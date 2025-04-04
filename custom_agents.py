import asyncio
import pandas as pd
import json
import logging
from enum import Enum
from pydantic import create_model
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Mapping, Sequence

from tqdm.auto import tqdm

from autogen_core import CancellationToken, Image, FunctionCall
from autogen_core.models import ChatCompletionClient
from autogen_core.models._types import SystemMessage, UserMessage

from autogen_agentchat.base import Response
from autogen_core.tools import FunctionTool, Tool
from autogen_agentchat.base import Handoff as HandoffBase

from autogen_agentchat.base import TaskResult, Team
from autogen_agentchat.messages import (
    #AgentMessage,
    ChatMessage,
    HandoffMessage,
    MultiModalMessage,
    StopMessage,
    TextMessage,
    #ToolCallMessage,
    #ToolCallResultMessage,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_agentchat.state import AssistantAgentState
from autogen_agentchat.agents._base_chat_agent import BaseChatAgent
from autogen_agentchat.agents._assistant_agent import AssistantAgent

from autogen_agentchat import EVENT_LOGGER_NAME
event_logger = logging.getLogger(EVENT_LOGGER_NAME)

class MyAgent(BaseChatAgent):
    def __init__(
            self,
            name: str,
            model_client: ChatCompletionClient,
            description: str,
            profile: str,
            reflection_task: str,
            response_task: str,
            pbar: tqdm,
            questionnaire_data: list | None = None,
            questionnaire_repetitions: int = 10,
            # TODO add a semi-constant seed!!
    ) -> None:
        super().__init__(name=name, description=description)
        self._model_client = model_client
        self._messsage_history = []
        self._thoughts = []
        self._questionnaire_responses = []
        self._profile = profile + '\n'
        self._reflection_instr = reflection_task
        self._response_instr = response_task.format(name=name, profile=profile) + '\n'
        self._questionnaire_data = questionnaire_data
        self._questionnaire_repetitions = questionnaire_repetitions
        self._pbar = pbar
        self._round = 1 # how long has the conversation been ongoing?
    
    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        return
    
    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:

        for msg in messages:
            if msg.source != "user": # exclude the opening task
                self._messsage_history.append(msg)
        
        ## Simulate the reflection
        print('round', self._round, self._name, 'thinking...')
        input_msg = self._reflection_instr.format(name = self.name,
                                                  profile = self._profile,
                                                  transcript = self._create_transcript(self._messsage_history[-5:])) #TODO: adjust maximal history
        reflection = await self._model_client.create([UserMessage(content=input_msg, source='user')],
                                                 cancellation_token=cancellation_token)
        self._thoughts.append(reflection)
        
        ## Fill out the questionnaire, if provided
        if self._questionnaire_data is not None:
            print('round', self._round, self._name, 'filling out the questionnaire...')
            questionnaire_prefix = f'You are {self._name}.\n{self._profile}'
            questionnaire_prefix += self._create_transcript(self._messsage_history[-5:])
            questionnaire_prefix += f'YOUR REFLECTION: {reflection.content}\n'

            #questionnaire_response = await self._model_client.create([UserMessage(content=input_msg, source='user')],
            #                                        cancellation_token=cancellation_token)
            for section in self._questionnaire_data:
    
                ## Construct the local text context from the JSON
                input_msg = questionnaire_prefix
                if 'introduction' in section.keys():
                    input_msg += section['introduction'] + '\n'
                else:
                    input_msg += 'Please respond to the following questionnaire:\n'
                input_msg += '\n'.join([question['id'] + ': ' + question['text'] for question in section['questions']]) + '\n'

                # open ended responses should be unstructured
                if 'options' in section['questions'][0]:
                    input_msg += (
                        'Answer in the following JSON format: {' + 
                        ', '.join([f"'{question['id']}': your response" for question in section['questions']]) + 
                        '}'
                    )
                    input_msg += (
                        '\nWhere your response is from one of the following options: ' +
                        str(section['questions'][0]['options']) + 
                        '\nDo not provide any additional explanation, just one of the possible answer options.'
                    )

                ## Construct the dynamic pydantic model
                # This code does not produce duplicate enums after conversion
                DynamicEnums = [Enum('DynamicEnum', {option: option for option in question['options']})
                                if 'options' in question else str # handle questions without answer options as open ended
                                for question in section['questions']]
                q_ids = [question['id'] for question in section['questions']]
                answer_types = {q_id: (enum, ...) for q_id, enum in zip(q_ids, DynamicEnums)}
                #answer_types.update({'explanation': (str, ...)}) # OPTIONAL: add an additional explanation after each section
                DynamicAnswer = create_model("DynamicAnswer", **answer_types)

                # TODO: parallelize this!
                for repetition in range(self._questionnaire_repetitions):

                    # only use structured outputs with closed-ended questions
                    if 'options' in section['questions'][0]:
                        extra_create_args = {"response_format": DynamicAnswer}
                    else:
                        extra_create_args = {}

                    response = await self._model_client.create(
                        [UserMessage(content=input_msg, source='user')],
                        extra_create_args = extra_create_args
                        )
                    
                    # parse structured response to closed-ended questions
                    if 'options' in section['questions'][0]:
                        result = json.loads(response.content)
                    else:
                        result = {'11.2': response.content} # TODO: fix this to be the correct question id
                    result['round'] = self._round
                    result['repetition'] = repetition
                    self._questionnaire_responses.append(result)

        ## Create a response
        print('round', self._round, self._name, 'creating answer...')
        input_msg = self._response_instr + self._create_transcript(self._messsage_history[-5:])
        input_msg += f'YOUR REFLECTION: {reflection.content}'
        #print(f'---------- {self._name} reflection ----------')
        #print(input_msg)
        result = await self._model_client.create([UserMessage(content=input_msg, source='user')],
                                                 cancellation_token=cancellation_token)
        response = Response(
            chat_message=TextMessage(content=result.content, source=self.name, models_usage=result.usage),
            inner_messages=[],
        )
        self._messsage_history.append(response.chat_message)
        self._round += 1
        self._pbar.update(1)

        return response
    
    def _create_transcript(self, messages: Sequence[ChatMessage]) -> str:
        transcript = "THE CONVERSATION SO FAR:\n"
        for message in messages:
            if isinstance(message, TextMessage | StopMessage | HandoffMessage):
                transcript += f"{message.source}: {message.content}\n"
            else:
                raise ValueError(f"Unexpected message type: {message} in {self.__class__.__name__}")
        return transcript

    async def _create_opener(self, task) -> TextMessage:
        print(self._name, 'creating opening statement...')
        task_msg = self._profile + task
        result = await self._model_client.create([UserMessage(content=task_msg, source='user')])
        return TextMessage(content=result.content, source=self.name, models_usage=result.usage)







class MySocietyOfMindAgent(BaseChatAgent):
    def __init__(
        self,
        name: str,
        team: Team,
        model_client: ChatCompletionClient,
        *,
        description: str = "An agent that uses an inner team of agents to generate responses.",
        task_prompt: str = "{transcript}\nContinue.",
        response_prompt: str = "Here is a transcript of conversation so far:\n{transcript}\n\\Provide a response to the original request.",
    ) -> None:
        super().__init__(name=name, description=description)
        self._team = team
        self._model_client = model_client
        if "{transcript}" not in task_prompt:
            raise ValueError("The task prompt must contain the '{transcript}' placeholder for the transcript.")
        self._task_prompt = task_prompt
        if "{transcript}" not in response_prompt:
            raise ValueError("The response prompt must contain the '{transcript}' placeholder for the transcript.")
        self._response_prompt = response_prompt

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        # Call the stream method and collect the messages.
        response: Response | None = None
        async for msg in self.on_messages_stream(messages, cancellation_token):
            if isinstance(msg, Response):
                response = msg
        assert response is not None
        return response

    async def on_messages_stream(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[ChatMessage | Response, None]:
        # Build the context.
        delta = list(messages)
        task: str | None = None
        if len(delta) > 0:
            task = self._task_prompt.format(transcript=self._create_transcript(delta))

        # Run the team of agents.
        result: TaskResult | None = None
        inner_messages: List[LLMMessage] = []
        async for inner_msg in self._team.run_stream(task=task, cancellation_token=cancellation_token):
            if isinstance(inner_msg, TaskResult):
                result = inner_msg
            else:
                yield inner_msg
                inner_messages.append(inner_msg)
        assert result is not None

        if len(inner_messages) < 2:
            # The first message is the task message so we need at least 2 messages.
            yield Response(
                chat_message=TextMessage(source=self.name, content="No response."), inner_messages=inner_messages
            )
        else:
            prompt = self._response_prompt.format(transcript=self._create_transcript(inner_messages))#(inner_messages[1:]))
            print('---', prompt)
            completion = await self._model_client.create(
                #messages=[SystemMessage(content=prompt)], cancellation_token=cancellation_token
                messages=[UserMessage(source='user', content=prompt)], cancellation_token=cancellation_token
            )
            #print('---', completion)
            assert isinstance(completion.content, str)
            yield Response(
                chat_message=TextMessage(source=self.name, content=completion.content, models_usage=completion.usage),
                inner_messages=inner_messages,
            )

        # Reset the team.
        await self._team.reset()

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        await self._team.reset()

    def _create_transcript(self, messages: Sequence[LLMMessage]) -> str:
        transcript = ""
        for message in messages:
            if isinstance(message, TextMessage | StopMessage | HandoffMessage):
                transcript += f"{message.source}: {message.content}\n"
            elif isinstance(message, MultiModalMessage):
                for content in message.content:
                    if isinstance(content, Image):
                        transcript += f"{message.source}: [Image]\n"
                    else:
                        transcript += f"{message.source}: {content}\n"
            else:
                raise ValueError(f"Unexpected message type: {message} in {self.__class__.__name__}")
        return transcript
    





class MyAssistantAgent(BaseChatAgent):
    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        *,
        tools: List[Tool | Callable[..., Any] | Callable[..., Awaitable[Any]]] | None = None,
        handoffs: List[HandoffBase | str] | None = None,
        description: str = "An agent that provides assistance with ability to use tools.",
        system_message: str
        | None = "You are a helpful AI assistant. Solve tasks using your tools. Reply with TERMINATE when the task has been completed.",
    ):
        super().__init__(name=name, description=description)
        self._model_client = model_client
        if system_message is None:
            self._system_messages = []
        else:
            self._system_messages = [SystemMessage(content=system_message)]
        self._tools: List[Tool] = []
        if tools is not None:
            if model_client.capabilities["function_calling"] is False:
                raise ValueError("The model does not support function calling.")
            for tool in tools:
                if isinstance(tool, Tool):
                    self._tools.append(tool)
                elif callable(tool):
                    if hasattr(tool, "__doc__") and tool.__doc__ is not None:
                        description = tool.__doc__
                    else:
                        description = ""
                    self._tools.append(FunctionTool(tool, description=description))
                else:
                    raise ValueError(f"Unsupported tool type: {type(tool)}")
        # Check if tool names are unique.
        tool_names = [tool.name for tool in self._tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError(f"Tool names must be unique: {tool_names}")
        # Handoff tools.
        self._handoff_tools: List[Tool] = []
        self._handoffs: Dict[str, HandoffBase] = {}
        if handoffs is not None:
            if model_client.capabilities["function_calling"] is False:
                raise ValueError("The model does not support function calling, which is needed for handoffs.")
            for handoff in handoffs:
                if isinstance(handoff, str):
                    handoff = HandoffBase(target=handoff)
                if isinstance(handoff, HandoffBase):
                    self._handoff_tools.append(handoff.handoff_tool)
                    self._handoffs[handoff.name] = handoff
                else:
                    raise ValueError(f"Unsupported handoff type: {type(handoff)}")
        # Check if handoff tool names are unique.
        handoff_tool_names = [tool.name for tool in self._handoff_tools]
        if len(handoff_tool_names) != len(set(handoff_tool_names)):
            raise ValueError(f"Handoff names must be unique: {handoff_tool_names}")
        # Check if handoff tool names not in tool names.
        if any(name in tool_names for name in handoff_tool_names):
            raise ValueError(
                f"Handoff names must be unique from tool names. Handoff names: {handoff_tool_names}; tool names: {tool_names}"
            )
        self._model_context: List[LLMMessage] = []
        self._is_running = False

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        """The types of messages that the assistant agent produces."""
        if self._handoffs:
            return [TextMessage, HandoffMessage]
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise AssertionError("The stream should have returned the final result.")

    async def on_messages_stream(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[LLMMessage | Response, None]:
        # Add messages to the model context.
        for msg in messages:
            if isinstance(msg, MultiModalMessage) and self._model_client.capabilities["vision"] is False:
                raise ValueError("The model does not support vision.")
            self._model_context.append(UserMessage(content=msg.content, source=msg.source))

        # Inner messages.
        inner_messages: List[LLMMessage] = []

        # Generate an inference result based on the current model context.
        llm_messages = self._system_messages + self._model_context
        result = await self._model_client.create(
            llm_messages, tools=self._tools + self._handoff_tools, cancellation_token=cancellation_token
        )

        # Add the response to the model context.
        self._model_context.append(AssistantMessage(content=result.content, source=self.name))

        # Run tool calls until the model produces a string response.
        while isinstance(result.content, list) and all(isinstance(item, FunctionCall) for item in result.content):
            #tool_call_msg = ToolCallMessage(content=result.content, source=self.name, models_usage=result.usage)
            #event_logger.debug(tool_call_msg)
            ## Add the tool call message to the output.
            #inner_messages.append(tool_call_msg)
            #yield tool_call_msg
#
            ## Execute the tool calls.
            #results = await asyncio.gather(
            #    *[self._execute_tool_call(call, cancellation_token) for call in result.content]
            #)
            #tool_call_result_msg = ToolCallResultMessage(content=results, source=self.name)
            #event_logger.debug(tool_call_result_msg)
            #self._model_context.append(FunctionExecutionResultMessage(content=results))
            #inner_messages.append(tool_call_result_msg)
            #yield tool_call_result_msg

            # Detect handoff requests.
            handoffs: List[HandoffBase] = []
            for call in result.content:
                if call.name in self._handoffs:
                    handoffs.append(self._handoffs[call.name])
            if len(handoffs) > 0:
                if len(handoffs) > 1:
                    raise ValueError(f"Multiple handoffs detected: {[handoff.name for handoff in handoffs]}")
                # Return the output messages to signal the handoff.
                yield Response(
                    chat_message=HandoffMessage(
                        content=handoffs[0].message, target=handoffs[0].target, source=self.name
                    ),
                    inner_messages=inner_messages,
                )
                return

            # Generate an inference result based on the current model context.
            llm_messages = self._system_messages + self._model_context
            result = await self._model_client.create(
                llm_messages, tools=self._tools + self._handoff_tools, cancellation_token=cancellation_token
            )
            self._model_context.append(AssistantMessage(content=result.content, source=self.name))

        assert isinstance(result.content, str)
        yield Response(
            chat_message=TextMessage(content=result.content, source=self.name, models_usage=result.usage),
            inner_messages=inner_messages,
        )

    async def _execute_tool_call(
        self, tool_call: FunctionCall, cancellation_token: CancellationToken
    ) -> FunctionExecutionResult:
        """Execute a tool call and return the result."""
        try:
            if not self._tools + self._handoff_tools:
                raise ValueError("No tools are available.")
            tool = next((t for t in self._tools + self._handoff_tools if t.name == tool_call.name), None)
            if tool is None:
                raise ValueError(f"The tool '{tool_call.name}' is not available.")
            arguments = json.loads(tool_call.arguments)
            result = await tool.run_json(arguments, cancellation_token)
            result_as_str = tool.return_value_as_string(result)
            return FunctionExecutionResult(content=result_as_str, call_id=tool_call.id)
        except Exception as e:
            return FunctionExecutionResult(content=f"Error: {e}", call_id=tool_call.id)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the assistant agent to its initialization state."""
        self._model_context.clear()

    async def save_state(self) -> Mapping[str, Any]:
        """Save the current state of the assistant agent."""
        return AssistantAgentState(llm_messages=self._model_context.copy()).model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        """Load the state of the assistant agent"""
        assistant_agent_state = AssistantAgentState.model_validate(state)
        self._model_context.clear()
        self._model_context.extend(assistant_agent_state.llm_messages)