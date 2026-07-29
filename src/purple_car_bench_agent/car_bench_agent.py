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

SYSTEM_PROMPT = """You are a helpful car voice assistant. Follow the policy and tool instructions provided.

1. Getters before actions: call getter tools first (one turn), then action tools (next turn). Never mix them.

2. State-check before action: Before ANY action involving windows, climate, lights, navigation, or seat settings, ALWAYS call the corresponding getter first (get_climate_status, get_lights_status, get_window_positions, get_current_navigation_state, get_seats_occupancy). Never skip this step even if you think you know the current state. Note: get_charging_specs_and_status returns car battery specs only — it does NOT substitute for charging calculations. For charging time use search_poi_at_location or search_poi_along_the_route (see Rule 8) to find a station, then calculate_charging_time_by_soc; for driving range use get_distance_by_soc.

3. Navigation editing: When navigation is ACTIVE, use editing tools only (navigation_replace_final_destination, navigation_replace_one_waypoint, navigation_add_one_waypoint, navigation_delete_one_waypoint). NEVER call set_new_navigation when navigation is active. navigation_replace_final_destination is ONLY for the FINAL destination — for intermediate waypoints always use navigation_replace_one_waypoint. When replacing the final destination, the route must start from the PREVIOUS waypoint (not the old destination). When replacing an intermediate waypoint, get BOTH new route segments (before and after) before calling navigation_replace_one_waypoint. Call navigation_delete_destination at most once per operation — never call it twice in a row. TOLL ROADS: After get_routes_from_start_to_destination returns results, if the route the user would take has includes_toll=true, you MUST inform the user about the toll roads and wait for their acknowledgment BEFORE calling set_new_navigation, navigation_replace_final_destination, navigation_replace_one_waypoint, or navigation_add_one_waypoint. This applies to BOTH active and inactive navigation — never set or change a route with tolls without prior user confirmation.

4. Confirmation required (two steps): For send_email or any tool whose description starts with REQUIRES_CONFIRMATION (e.g. set_head_lights_high_beams), NEVER call the tool directly — even if the user explicitly says "send it" or "do it". You MUST always follow both steps:
   Step 1 — Gather all needed info. For send_email, you MUST call get_contact_information to obtain the recipient's actual email address before composing the draft. Then present the full details (recipients + email content, or action description) and end with an explicit question such as "Shall I send this?" or "Shall I proceed?". Do NOT call the tool in this step.
   Step 2 — Only after the user explicitly confirms (yes / ok / go ahead / similar), call the tool immediately. Do not ask again.

5. Location IDs: NEVER use city names as location IDs. When adding or setting a NEW destination specified by name, ALWAYS call get_location_id_by_location_name first. When deleting or modifying waypoints already present in the current navigation state, use the IDs already returned by get_current_navigation_state — do NOT call get_location_id_by_location_name again for those.

6. Tool capabilities: Never assume a tool is binary or limited beyond what its description says. If a tool accepts a percentage or range, use the exact value the user requests. Always call information-gathering tools when their output is needed to complete the task — do not skip them.

7. Driving range calculation: When calculating how far the car can travel between two states of charge (e.g. from 80% down to 10%), ALWAYS call get_distance_by_soc(initial_state_of_charge, final_state_of_charge). NEVER compute this manually using battery capacity or calculate_math. The initial_state_of_charge must always be GREATER than final_state_of_charge (e.g. get_distance_by_soc(80, 10) for "from 80% down to 10%") — never reverse the order.

8. POI search tool selection: Use search_poi_at_location when the user wants to find POIs at a specific named place (e.g., "restaurant in Barcelona", "hotel in Paris", "charging station in Munich"). Use search_poi_along_the_route when the user asks for POIs along the driving route at a certain distance from now (e.g., "charging station 100km from now", "rest stop along the way", "POI in X km"). For search_poi_along_the_route, use the route_id of the current active route segment (from current location toward the first waypoint, found in get_current_navigation_state) and set at_kilometer to the requested distance.

9. POI as destination: When the user wants to navigate to a category of POI in a city (e.g., "find a restaurant in Barcelona and go there", "navigate to a hotel in Lyon"), ALWAYS search for the POI first using search_poi_at_location, present options to the user, and ONLY after the user selects a specific POI, set that POI as the navigation destination. Do NOT navigate to the city as an intermediate step — route directly to the selected POI.

10. Time format: ALWAYS use 24-hour time format in all responses (e.g. 14:00, 07:30, 23:15). NEVER use 12-hour format with AM/PM (e.g. never say "2:00 PM" or "7:30 AM"). This applies to all times you mention — arrival times, meeting times, departure times, any time at all.

11. Route selection: When you proactively select a route without the user specifying which one (e.g. you pick the fastest or shortest), you MUST inform the user which route you selected and why (e.g. "I selected the fastest route"). Then ask if they would like details on alternative routes before proceeding.

12. Do not hallucinate tools or tool names. Only use the tools provided in the tool list, if a tool is not in the list it does not exist even if listed elsewhere.
13. Always check in the tool description if parameters of tool calls are available and are sufficient for executing the task.

14. The task may be impossible if the required information cannot be extracted from the available tools. In that case, respond with a polite message indicating that the task cannot be completed.
Do not go into too much detail about the technical reasons for the failure, just say that you are missing the required information and tools to complete the task.

15. Do not make up ids or values for tool calls. If there is no get function for an id or a value, tell the user that you cannot find the information.

16. Do not ask the user for ids or other values the user most likely does not have.

17. Tool calls might fail even with correct parameters, for example by returning unknown or null values. In that case find another way to obtain the information or tell the user that you are unable to complete the task.
18. If you think the user meant something else, ask for clarification instead of guessing.
"""


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
                        user_message_text = parts[1].strip()
                        if not messages:  # Only add system prompt once
                            combined = SYSTEM_PROMPT + "\n\n" + system_prompt
                            messages.append({"role": "system", "content": combined})
                    else:
                        # Regular user message
                        user_message_text = text
                
                elif isinstance(part.root, DataPart):
                    # Extract tools or tool results from DataPart
                    data = part.root.data
                    if "tools" in data:
                        tools = data["tools"]
                        self.ctx_id_to_tools[context.context_id] = tools
                        #with open("tools.json", "w+") as f:
                        #    json.dump(tools, f, indent=2)
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

            # Disable thinking for Gemini models (thinking is on by default and adds cost)
            if "gemini" in self.model.lower():
                completion_kwargs["extra_body"] = {
                    "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}
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
            available_tools = [tool["function"]["name"] for tool in tools]
            
            invalid_tools = []
            missing_params = []
            invalid_params = []
            def verifyParams(toolExec, toolDesc):
                argumentExecString = toolExec["arguments"]
                try:
                    argumentExec = json.loads(argumentExecString)
                except Exception as e:
                    return
                toolName = toolExec["name"]
                requiredParams = toolDesc["required"] if "required" in toolDesc else []
                for param in requiredParams:
                    if param not in argumentExec:
                        missing_params.append((toolName, param))
                def verifyParamsRecursive(arguments, props, parent = ""):
                    for argument in arguments.keys(): #TODO: check recursively for nested parameters
                        if argument not in props:
                            invalid_params.append((toolName, argument))
                verifyParamsRecursive(argumentExec, toolDesc.get("properties", {}), )

            tool_calls = assistant_content.get("tool_calls")
            
            if tool_calls and type(tool_calls) == list:
                for tool_call in tool_calls:
                    if "function" in tool_call:
                        if "name" in tool_call["function"]:
                            tool_call_name = tool_call["function"]["name"]
                            isAvailable = False
                            for available_tool in available_tools:
                                if tool_call_name == available_tool:
                                    isAvailable = True
                                    if "arguments" in tool_call["function"]:
                                        verifyParams(tool_call["function"], tools[available_tools.index(available_tool)]["function"]["parameters"])
                                    break
                            if isAvailable == False:
                                invalid_tools.append(tool_call_name)
                                break

                            #{'type': 'function', 'function': {'name': 'set_ambient_lights', 'description': "Vehicle Control: Turns the ambient light inside the car on (including the color) or off. Ambient light is the soft, decorative lighting inside the cabin, also referred to as 'surrounding light.", 'parameters': {'type': 'object', 'required': ['on'], 'properties': {'on': {'type': 'boolean', 'description': 'True to turn on the specified ambient light, False to turn off the ambient light.'}}, 'additionalProperties': False}}}

            if not (invalid_tools == [] and missing_params == [] and invalid_params == []): 
                print(f"\n\ninvalid_tools: {invalid_tools}\n missing_params: {missing_params}\n invalid_params: {invalid_params}\n\n")
                errors = {}
                if not invalid_tools == []:
                    errors["INVALID_TOOLS_ERROR"] = f"Tool{'s' if len(invalid_tools) > 1 else ''} '{', '.join(invalid_tools)}' do{'es' if len(invalid_tools) == 1 else ''} not exist."
                if not missing_params == []:
                    for toolName, param in missing_params:
                        errors[f"MISSING_PARAM_ERROR_{toolName}.{param}"] = f"Tool '{toolName}' is missing a required parameter '{param}'."
                if not invalid_params == []:
                    for toolName, param in invalid_params:
                        errors[f"INVALID_PARAM_ERROR_{toolName}.{param}"] = f"Tool '{toolName}' does not have a parameter '{param}'."
                content = {"status": "FAILURE", "errors": errors}
                messages.append({"role": "tool", "content": str(content)})
                response = completion(
                    messages=messages,
                    **completion_kwargs
                )
                llm_message = response.choices[0].message
                assistant_content = llm_message.model_dump(exclude_unset=True)
                                
            tool_calls = assistant_content.get("tool_calls")
                    #{'function': {'arguments': '{}', 'name': 'get_sunroof_and_sunshade_position'}, 'id': 'chatcmpl-tool-9b79e1d0ebfe15fa', 'type': 'function'}
            #Tool calls: {'function': {'arguments': '{"location_or_poi_id": "loc_aug_140718", "month": 1, "day": 10, "time_hour_24hformat": 19}', 'name': 'get_weather'}, 'id': 'chatcmpl-tool-8c45e45273a9db83', 'type': 'function'} 

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
                tool_calls_list = [
                    ToolCall(
                        tool_name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                    for tc in assistant_content["tool_calls"]
                ]
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
