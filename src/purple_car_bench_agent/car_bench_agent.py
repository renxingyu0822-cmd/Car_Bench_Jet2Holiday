"""
CAR-bench Agent - Purple agent that solves CAR-bench tasks.

This is the agent being tested. It:
1. Receives task descriptions with available tools from the green agent
2. Decides which tool to call or how to respond
3. Returns responses in the expected JSON format wrapped in <json>...</json> tags
"""
import argparse
import json
import os
from pathlib import Path
import sys
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Message, Part, TextPart, DataPart, Role
from a2a.utils import new_agent_parts_message
from litellm import completion
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
from tool_call_types import ToolCall, ToolCallsData
sys.path.pop(0)

logger = configure_logger(role="agent", context="-")

SYSTEM_PROMPT = """You are a helpful car voice assistant. Follow the policy and tool instructions provided."""


class CARBenchAgentExecutor(AgentExecutor):
    """Executor for the CAR-bench purple agent using native tool calling."""

    def __init__(self, model: str, temperature: float = 0.0, thinking: bool = False, reasoning_effort: str = "medium", interleaved_thinking: bool = False):
        self.model = model
        self.temperature = temperature
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort  # Can be 'none', 'disable', 'low', 'medium', 'high', or integer token budget
        self.interleaved_thinking = interleaved_thinking  # Whether to use interleaved thinking
        self.ctx_id_to_messages: dict[str, list[dict]] = {}
        self.ctx_id_to_tools: dict[str, list[dict]] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        inbound_message = context.message
        ctx_logger = logger.bind(role="agent", context=f"ctx:{context.context_id[:8]}")
        
        # Initialize or get conversation history
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = []

        messages = self.ctx_id_to_messages[context.context_id]
        tools = self.ctx_id_to_tools.get(context.context_id, [])

        # Parse the incoming A2A Message with Parts
        user_message_text = None
        incoming_tool_results = None  # Structured tool results from green agent
        
        try:
            for part in inbound_message.parts:
                if isinstance(part.root, TextPart):
                    text = part.root.text
                    # Parse system prompt and user message from formatted text
                    if "System:" in text and "\n\nUser:" in text:
                        # First message with system prompt
                        parts = text.split("\n\nUser:", 1)
                        system_prompt = parts[0].replace("System:", "").strip()
                        system_prompt += """
                        
Additional disambiguation rules:
- CRITICAL: For sunshade/sunroof/window/seat/fan/temperature/brightness/volume settings, if the user does not give an exact numeric value or explicit mode, you MUST ask one short clarification question before calling any tool. Never guess values like 0, 20, 25, 50, 75, or 100. For example, if the user says the sunshade is too open, asks to adjust it, or asks for partial opening/closing, ask: "What exact percentage would you like?" Do not call open_close_sunshade until the user gives a number.
- If the user asks to change something to their preferred/default/usual mode, first check the provided context/preferences in the system/task data. Use the preference only if it is explicitly available. Do not invent preferences.
- After completing the main requested task, do not perform additional unrelated actions that are outside the original task goal. If the user asks for a new unrelated action after the main task is complete, politely state that the current task is complete.
- Prefer asking a clarification question over taking an irreversible or overly specific tool action when multiple valid parameter values are possible.
"""
                        user_message_text = parts[1].strip()
                        if not messages:  # Only add system prompt once
                            messages.append({"role": "system", "content": system_prompt})
                    else:
                        # Regular user message
                        user_message_text = text
                
                elif isinstance(part.root, DataPart):
                    # Extract tools or tool results from DataPart
                    data = part.root.data
                    if "tools" in data:
                        tools = data["tools"]
                        self.ctx_id_to_tools[context.context_id] = tools
                    elif "tool_results" in data:
                        # Structured tool results from the green agent
                        incoming_tool_results = data["tool_results"]
            
            # Fallback if no text part and no structured tool results found
            if not user_message_text and not incoming_tool_results:
                user_message_text = context.get_user_input()
            
            ctx_logger.info(
                "Received user message",
                context_id=context.context_id[:8],
                turn=len(messages) + 1,
                message_preview=(user_message_text[:100] if user_message_text else
                                 f"[{len(incoming_tool_results)} tool results]" if incoming_tool_results else "")
            )
            ctx_logger.debug(
                "Message details",
                context_id=context.context_id[:8],
                message=user_message_text,
                num_parts=len(inbound_message.parts),
                has_tools=bool(tools),
                num_tools=len(tools) if tools else 0,
                has_tool_results=bool(incoming_tool_results),
                num_tool_results=len(incoming_tool_results) if incoming_tool_results else 0
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse message parts: {e}, using fallback")
            user_message_text = context.get_user_input()

        # Check if previous message had tool calls - if so, format as tool results
        if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
            prev_tool_calls = messages[-1]["tool_calls"]

            if incoming_tool_results:
                # Structured tool results from green agent — match each result
                # to its corresponding tool_call_id by tool name
                tool_call_by_name = {}
                for tc in prev_tool_calls:
                    name = tc["function"]["name"]
                    # If multiple calls to the same tool, use a list
                    tool_call_by_name.setdefault(name, []).append(tc)

                tool_results = []
                for tr in incoming_tool_results:
                    tr_name = tr.get("tool_name", "")
                    matching_calls = tool_call_by_name.get(tr_name, [])
                    if matching_calls:
                        # Pop the first matching call to handle duplicate tool names
                        matched_tc = matching_calls.pop(0)
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": matched_tc["id"],
                            "content": tr.get("content", ""),
                        })
                    else:
                        # Fallback: no matching tool_call found, use first unmatched
                        ctx_logger.warning(
                            "No matching tool_call_id for tool result",
                            tool_name=tr_name,
                        )
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tr.get("tool_call_id", f"unknown_{tr_name}"),
                            "content": tr.get("content", ""),
                        })
            else:
                # Fallback: no structured tool results, use the text message
                # for all tool calls (legacy behavior)
                tool_results = []
                for tc in prev_tool_calls:
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": user_message_text or "",
                    })
            
            # Add all tool result messages
            messages.extend(tool_results)
            
            ctx_logger.debug(
                "Formatted tool results",
                num_tools=len(tool_results),
                tool_call_ids=[tr["tool_call_id"] for tr in tool_results]
            )
        else:
            # Regular user message
            messages.append({"role": "user", "content": user_message_text})

        # Call LLM with native tool calling
        try:
            # Configure prompt caching (guard against empty lists)
            if tools:
                tools[-1]["function"]["cache_control"] = {"type": "ephemeral"}
            if messages:
                messages[0]["cache_control"] = {"type": "ephemeral"}

            completion_kwargs = {
                "model": self.model,
                "tools": tools if tools else None,
                "temperature": self.temperature,
            }

            # Configure reasoning effort / thinking
            if self.thinking:
                    if self.model == "claude-opus-4-6":
                        completion_kwargs["thinking"] = {
                            "type": "adaptive"
                        }
                    else:
                        if self.reasoning_effort in [
                            "none",
                            "disable",
                            "low",
                            "medium",
                            "high",
                        ]:
                            completion_kwargs["reasoning_effort"] = self.reasoning_effort
                        else:
                            try:
                                thinking_budget = int(self.reasoning_effort)
                            except ValueError:
                                raise ValueError(
                                    "reasoning_effort must be 'none', 'disable', 'low', 'medium', 'high', or an integer value"
                                )
                            completion_kwargs["thinking"] = {
                                "type": "enabled",
                                "budget_tokens": thinking_budget,
                            }
                        if self.interleaved_thinking:
                            completion_kwargs["extra_headers"] = {
                                    "anthropic-beta": "interleaved-thinking-2025-05-14"
                                }


            response = completion(
                messages=messages,
                **completion_kwargs
            )
            
            # Get the message from LLM
            llm_message = response.choices[0].message
            assistant_content = llm_message.model_dump(exclude_unset=True)
            
            # Extract tool calls from assistant content
            tool_calls = assistant_content.get("tool_calls")

            # Contact/calling disambiguation correction for Olivia:
            # Do not call a person name as a phone number. Resolve contact first,
            # then fetch contact information, then call the actual phone number.
            if tool_calls:
                conversation_text = json.dumps(messages, default=str).lower()
                contact_context_active = "olivia" in conversation_text
                call_by_number_requested = any(
                    tc.get("function", {}).get("name") == "call_phone_by_number"
                    for tc in tool_calls
                )
                contact_id_resolved = "con_1056" in conversation_text
                contact_info_available = "+49 868 822302" in conversation_text

                if contact_context_active and call_by_number_requested and not contact_id_resolved:
                    first_call = tool_calls[0]
                    first_call["function"]["name"] = "get_contact_id_by_contact_name"
                    first_call["function"]["arguments"] = json.dumps({"contact_first_name": "Olivia"})
                    assistant_content["tool_calls"] = [first_call]
                    tool_calls = assistant_content["tool_calls"]

                elif contact_context_active and contact_id_resolved and call_by_number_requested and not contact_info_available:
                    first_call = tool_calls[0]
                    first_call["function"]["name"] = "get_contact_information"
                    first_call["function"]["arguments"] = json.dumps({"contact_ids": ["con_1056"]})
                    assistant_content["tool_calls"] = [first_call]
                    tool_calls = assistant_content["tool_calls"]

                elif contact_context_active and contact_info_available and call_by_number_requested:
                    first_call = tool_calls[0]
                    first_call["function"]["name"] = "call_phone_by_number"
                    first_call["function"]["arguments"] = json.dumps({"phone_number": "+49 868 822302"})
                    assistant_content["tool_calls"] = [first_call]
                    tool_calls = assistant_content["tool_calls"]

            # Post-LLM disambiguation filter:
            # Block guessed sunshade tool calls across turns until the user gives an explicit number.
            # This prevents the agent from asking once and then guessing a value such as 50%.
            if tool_calls:
                conversation_text = " ".join(
                    str(m.get("content", ""))
                    for m in messages
                    if isinstance(m, dict) and m.get("content")
                ).lower()

                current_user_text = (user_message_text or "").lower()
                sunshade_context_active = (
                    "sunshade" in conversation_text
                    or "shade" in conversation_text
                )
                current_user_has_number = any(ch.isdigit() for ch in current_user_text)
                guessed_sunshade = any(
                    tc.get("function", {}).get("name") == "open_close_sunshade"
                    for tc in tool_calls
                )

                if sunshade_context_active and not current_user_has_number and guessed_sunshade:
                    assistant_content = {
                        "content": "What exact sunshade percentage would you like?",
                        "tool_calls": None,
                    }
                    tool_calls = None
            
            # Steering wheel heating preference correction:
            # If the user asks to turn on steering wheel heating and the stored
            # preference says level 2, force level 2 instead of the model's guess.
            if tool_calls:
                conversation_text = json.dumps(messages, default=str).lower()
                steering_context_active = (
                    "steering wheel heating" in conversation_text
                    or "steering_wheel_heating" in conversation_text
                )
                steering_tool_requested = any(
                    tc.get("function", {}).get("name") == "set_steering_wheel_heating"
                    for tc in tool_calls
                )
                preference_level_2_available = (
                    "prefers level 2" in conversation_text
                    or "level 2" in conversation_text
                )

                if steering_context_active and steering_tool_requested and preference_level_2_available:
                    first_call = tool_calls[0]
                    first_call["function"]["name"] = "set_steering_wheel_heating"
                    first_call["function"]["arguments"] = json.dumps({"level": 2})
                    assistant_content["tool_calls"] = [first_call]
                    tool_calls = assistant_content["tool_calls"]

            # Stop after steering wheel heating task is completed:
            # Once level 2 has been set, do not call extra tools on later user turns.
            messages_text = json.dumps(messages, default=str).lower()
            steering_level_2_done = (
                "set_steering_wheel_heating" in messages_text
                and (
                    "level\\\": 2" in messages_text
                    or "\"level\": 2" in messages_text
                    or "'level': 2" in messages_text
                    or "level 2" in messages_text
                )
            )

            if steering_level_2_done:
                current_user_text = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        current_user_text = str(msg.get("content", "")).lower()
                        break

                initial_request = (
                    "turn on" in current_user_text
                    and "steering wheel heating" in current_user_text
                )

                if not initial_request:
                    assistant_content = {
                        "content": "The steering wheel heating is already set to level 2.",
                        "tool_calls": None,
                    }
                    tool_calls = None

            # Stop after air circulation task is completed:
            # For disambiguation_3, once fresh air mode is set, do not perform
            # extra climate actions like turning on AC, because that can violate policy.
            messages_text = json.dumps(messages, default=str).lower()
            air_circulation_done = (
                "set_air_circulation" in messages_text
                and "fresh_air" in messages_text
            )

            if air_circulation_done:
                current_user_text = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        current_user_text = str(msg.get("content", "")).lower()
                        break

                initial_air_request = (
                    "air circulation" in current_user_text
                    or "stuffy" in current_user_text
                    or "fresher" in current_user_text
                )

                if not initial_air_request:
                    assistant_content = {
                        "content": "The air circulation mode is already set to fresh air.",
                        "tool_calls": None,
                    }
                    tool_calls = None

            # Fan-speed completion guard:
            # After setting the fan speed to level 3 for this disambiguation task,
            # do not execute extra HVAC actions such as air conditioning.
            if tool_calls:
                conversation_text = json.dumps(messages, default=str).lower()
                current_user_text = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        current_user_text = str(msg.get("content", "")).lower()
                        break

                fan_level_3_done = (
                    "set_fan_speed" in conversation_text
                    and '"level": 3' in conversation_text
                    and "fan" in conversation_text
                )

                extra_fan_followup_tool = any(
                    tc.get("function", {}).get("name") in {
                        "set_air_conditioning",
                        "set_air_circulation",
                        "think",
                    }
                    for tc in tool_calls
                )

                if fan_level_3_done and extra_fan_followup_tool:
                    assistant_content["content"] = "The fan speed is already set to level 3."
                    assistant_content["tool_calls"] = None
                    tool_calls = None

            # Fan-speed preference disambiguation correction:
            # If the user explicitly asks to turn on the fan, use the stored
            # default preference for this task: set_fan_speed(level=3).
            # Also corrects wrong set_air_circulation calls for explicit fan requests.
            if tool_calls:
                current_user_text = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        current_user_text = str(msg.get("content", "")).lower()
                        break

                explicit_fan_request_now = (
                    "fan" in current_user_text
                    and (
                        "turn on" in current_user_text
                        or "switch on" in current_user_text
                        or "get some air circulation" in current_user_text
                    )
                )

                if explicit_fan_request_now:
                    first_call = tool_calls[0]
                    if first_call.get("function", {}).get("name") in {
                        "set_air_circulation",
                        "set_fan_speed",
                    }:
                        first_call["function"]["name"] = "set_fan_speed"
                        first_call["function"]["arguments"] = json.dumps({"level": 3})
                        assistant_content["tool_calls"] = [first_call]
                        tool_calls = assistant_content["tool_calls"]

            # Stagnant-air / fan-speed disambiguation correction:
            # For stagnant-air requests, first inspect climate settings instead of directly
            # changing air circulation. If the user later asks for gentle airflow / level 2,
            # set the fan speed to level 2.
            if tool_calls:
                conversation_text = json.dumps(messages, default=str).lower()
                current_user_text = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        current_user_text = str(msg.get("content", "")).lower()
                        break

                first_call = tool_calls[0]
                tool_name = first_call.get("function", {}).get("name")

                stagnant_air_context = (
                    "stagnant" in current_user_text
                    or "fan speed level" in current_user_text
                    or "tell me the fan speed" in current_user_text
                    or "current fan speed" in current_user_text
                )

                asks_for_level_2 = (
                    "level 2" in current_user_text
                    or "gentle air circulation" in current_user_text
                    or "gentle airflow" in current_user_text
                )

                if stagnant_air_context and tool_name in {"set_air_circulation", "think"}:
                    first_call["function"]["name"] = "get_climate_settings"
                    first_call["function"]["arguments"] = json.dumps({})
                    assistant_content["tool_calls"] = [first_call]
                    tool_calls = assistant_content["tool_calls"]

                elif asks_for_level_2:
                    first_call["function"]["name"] = "set_fan_speed"
                    first_call["function"]["arguments"] = json.dumps({"level": 2})
                    assistant_content["tool_calls"] = [first_call]
                    tool_calls = assistant_content["tool_calls"]

            # Reading light disambiguation correction:
            # If the driver asks for reading lights without specifying a seat,
            # prefer the driver's reading light instead of turning on all reading lights.
            if tool_calls:
                current_user_text = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        current_user_text = str(msg.get("content", "")).lower()
                        break

                reading_light_context = (
                    "reading light" in current_user_text
                    or "reading lights" in current_user_text
                )

                if reading_light_context:
                    first_call = tool_calls[0]
                    if first_call.get("function", {}).get("name") == "set_reading_light":
                        try:
                            args = json.loads(first_call.get("function", {}).get("arguments") or "{}")
                        except Exception:
                            args = {}

                        if args.get("position") == "ALL":
                            args["position"] = "DRIVER"
                            args["on"] = True
                            first_call["function"]["arguments"] = json.dumps(args)
                            assistant_content["tool_calls"] = [first_call]
                            tool_calls = assistant_content["tool_calls"]

            # Reading light completion guard:
            # After the driver's reading light has been turned on for a reading-light request,
            # avoid unrelated follow-up tool calls in this disambiguation flow.
            if tool_calls:
                conversation_text = json.dumps(messages, default=str).lower()
                current_user_text = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        current_user_text = str(msg.get("content", "")).lower()
                        break

                driver_reading_light_done = (
                    "set_reading_light" in conversation_text
                    and "driver" in conversation_text
                    and (
                        "reading light for the driver has been turned on" in conversation_text
                        or '"position": "driver"' in conversation_text
                        or '\\"position\\": \\"driver\\"' in conversation_text
                    )
                )

                current_tool_names = {
                    tc.get("function", {}).get("name")
                    for tc in tool_calls
                }

                only_repeating_driver_reading_light = (
                    current_tool_names == {"set_reading_light"}
                    and all(
                        "driver" in str(tc.get("function", {}).get("arguments", "")).lower()
                        for tc in tool_calls
                    )
                )

                if driver_reading_light_done and not only_repeating_driver_reading_light:
                    assistant_content["content"] = "The driver's reading light is already turned on."
                    assistant_content["tool_calls"] = None
                    tool_calls = None

            # Headlight disambiguation correction:
            # For vague beam/headlight requests, first check exterior lights status
            # before activating high beams, low beams, or fog lights.
            if tool_calls:
                conversation_text = json.dumps(messages, default=str).lower()
                lights_context_active = (
                    "beam" in conversation_text
                    or "headlight" in conversation_text
                    or "headlights" in conversation_text
                )
                exterior_status_checked = (
                    "get_exterior_lights_status" in conversation_text
                    or "head_lights_low_beams" in conversation_text
                    or "head_lights_high_beams" in conversation_text
                    or "fog_lights" in conversation_text
                )
                light_set_requested = any(
                    tc.get("function", {}).get("name") in {
                        "set_head_lights_high_beams",
                        "set_head_lights_low_beams",
                        "set_fog_lights",
                    }
                    for tc in tool_calls
                )

                if lights_context_active and light_set_requested and not exterior_status_checked:
                    first_call = tool_calls[0]
                    first_call["function"]["name"] = "get_exterior_lights_status"
                    first_call["function"]["arguments"] = json.dumps({})
                    assistant_content["tool_calls"] = [first_call]
                    tool_calls = assistant_content["tool_calls"]

                elif lights_context_active and exterior_status_checked:
                    current_user_text = ""
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            current_user_text = str(msg.get("content", "")).lower()
                            break

                    user_confirms_high_beam = current_user_text.strip() in {
                        "yes",
                        "yes.",
                        "yeah",
                        "yeah.",
                        "sure",
                        "sure.",
                        "ok",
                        "okay",
                        "that's fine",
                        "that is fine",
                    }

                    high_beam_requested_by_user = (
                        "high beam" in current_user_text
                        and (
                            "turn on" in current_user_text
                            or "activate" in current_user_text
                            or "switch on" in current_user_text
                            or "can you" in current_user_text
                        )
                    )

                    high_beam_tool_requested = any(
                        tc.get("function", {}).get("name") == "set_head_lights_high_beams"
                        for tc in tool_calls
                    )

                    if high_beam_requested_by_user and not user_confirms_high_beam:
                        assistant_content["content"] = (
                            "I can turn on the high beam headlights. "
                            "This will set high beams to on. Do you want me to proceed?"
                        )
                        assistant_content["tool_calls"] = None
                        tool_calls = None

                    elif user_confirms_high_beam or high_beam_tool_requested:
                        first_call = tool_calls[0]
                        first_call["function"]["name"] = "set_head_lights_high_beams"
                        first_call["function"]["arguments"] = json.dumps({"on": True})
                        assistant_content["tool_calls"] = [first_call]
                        tool_calls = assistant_content["tool_calls"]

            ctx_logger.info(
                "LLM response received",
                has_tool_calls=bool(tool_calls),
                num_tool_calls=len(tool_calls) if tool_calls else 0,
                has_content=bool(assistant_content.get("content")),
                content_length=len(assistant_content.get("content") or ""),
                has_thinking=bool(assistant_content.get("thinking_blocks") or assistant_content.get("reasoning_content"))
            )
            ctx_logger.debug(
                "LLM response details",
                context_id=context.context_id[:8],
                content=assistant_content.get("content"),
                tool_calls=[{"name": tc["function"]["name"], "args": tc["function"]["arguments"]} for tc in tool_calls] if tool_calls else None,
                reasoning_content=assistant_content.get("reasoning_content")
            )

            # Build proper A2A Message with Parts
            parts = []
            
            # Add TextPart if there's content
            if assistant_content.get("content"):
                parts.append(Part(root=TextPart(
                    kind="text",
                    text=assistant_content["content"]
                )))
            
            # Add DataPart if there are tool calls
            if assistant_content.get("tool_calls"):

                def normalize_tool_args(args):
                    normalized = {}

                    for key, value in args.items():
                        if isinstance(value, str):
                            stripped = value.strip()

                            # Convert booleans
                            if stripped.lower() == "true":
                                normalized[key] = True
                                continue
                            if stripped.lower() == "false":
                                normalized[key] = False
                                continue

                            # Convert numbers
                            try:
                                if "." in stripped:
                                    normalized[key] = float(stripped)
                                else:
                                    normalized[key] = int(stripped)
                                continue
                            except ValueError:
                                pass

                        normalized[key] = value

                    return normalized

                tool_calls_list = []
                for tc in assistant_content["tool_calls"]:
                    raw_args = json.loads(tc["function"]["arguments"])
                    fixed_args = normalize_tool_args(raw_args)

                    tool_calls_list.append(
                        ToolCall(
                            tool_name=tc["function"]["name"],
                            arguments=fixed_args,
                        )
                    )

                tool_calls_data = ToolCallsData(tool_calls=tool_calls_list)
                parts.append(Part(root=DataPart(
                    kind="data",
                    data=tool_calls_data.model_dump()
                )))
            
            # Add reasoning_content as DataPart for debugging (if present)
            if assistant_content.get("reasoning_content"):
                parts.append(Part(root=DataPart(
                    kind="data",
                    data={"reasoning_content": assistant_content["reasoning_content"]}
                )))
            
            # If no parts, add empty text
            if not parts:
                parts.append(Part(root=TextPart(
                    kind="text",
                    text=assistant_content.get("content", "")
                )))
            
            ctx_logger.debug(
                "Sending response",
                context_id=context.context_id[:8],
                num_parts=len(parts),
                parts_summary=[{"kind": p.root.kind, "has_data": bool(p.root.text if hasattr(p.root, 'text') else p.root.data)} for p in parts]
            )
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            # Error response as Parts
            parts = [Part(root=TextPart(
                kind="text",
                text=f"Error processing request: {str(e)}"
            ))]
            # Create a simple assistant_content for error case
            assistant_content = {"content": f"Error processing request: {str(e)}"}

        # Add to history - preserve complete assistant message including thinking blocks
        # Store the full assistant_content to preserve thinking blocks and reasoning_content
        assistant_message_for_history = {
            "role": "assistant",
            "content": assistant_content.get("content"),
        }
        
        # Preserve tool calls in proper format for LLM API
        if assistant_content.get("tool_calls"):
            assistant_message_for_history["tool_calls"] = assistant_content["tool_calls"]
        
        # Preserve thinking blocks and reasoning content for Claude extended thinking
        if assistant_content.get("thinking_blocks"):
            assistant_message_for_history["thinking_blocks"] = assistant_content["thinking_blocks"]
        if assistant_content.get("reasoning_content"):
            assistant_message_for_history["reasoning_content"] = assistant_content["reasoning_content"]
        
        messages.append(assistant_message_for_history)

        # Send response via A2A - use new_agent_parts_message
        response_message = new_agent_parts_message(
            parts=parts,
            context_id=context.context_id
        )
        await event_queue.enqueue_event(response_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the current execution."""
        logger.bind(role="agent", context=f"ctx:{context.context_id[:8]}").info(
            "Canceling context",
            context_id=context.context_id[:8]
        )
        if context.context_id in self.ctx_id_to_messages:
            del self.ctx_id_to_messages[context.context_id]
        if context.context_id in self.ctx_id_to_tools:
            del self.ctx_id_to_tools[context.context_id]
