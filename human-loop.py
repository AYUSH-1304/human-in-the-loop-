import os
from uuid import uuid4
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY is missing from .env"
    )

if not os.getenv("TAVILY_API_KEY"):
    raise ValueError(
        "TAVILY_API_KEY is missing from .env"
    )


# ============================================================
# 2. In-memory checkpointer
# ============================================================

memory = InMemorySaver()


# ============================================================
# 3. Custom message reducer
# ============================================================

def reduce_messages(
    left: list[AnyMessage],
    right: list[AnyMessage]
) -> list[AnyMessage]:

    # Assign IDs to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())

    # Copy existing messages
    merged = left.copy()

    # Replace messages with same ID
    # Otherwise append new messages
    for message in right:

        for i, existing in enumerate(merged):

            if existing.id == message.id:
                merged[i] = message
                break

        else:
            merged.append(message)

    return merged


# ============================================================
# 4. Agent state
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[
        list[AnyMessage],
        reduce_messages
    ]


# ============================================================
# 5. Tavily search tool
# ============================================================

tool = TavilySearch(
    max_results=2
)


# ============================================================
# 6. Agent
# ============================================================

class Agent:

    def __init__(
        self,
        model,
        tools,
        system="",
        checkpointer=None
    ):

        self.system = system

        # ----------------------------------------------------
        # Create graph
        # ----------------------------------------------------

        graph = StateGraph(AgentState)

        graph.add_node(
            "llm",
            self.call_model
        )

        graph.add_node(
            "action",
            self.take_action
        )

        # ----------------------------------------------------
        # LLM -> action OR END
        # ----------------------------------------------------

        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {
                True: "action",
                False: END
            }
        )

        # ----------------------------------------------------
        # action -> LLM
        # ----------------------------------------------------

        graph.add_edge(
            "action",
            "llm"
        )

        # ----------------------------------------------------
        # Entry point
        # ----------------------------------------------------

        graph.set_entry_point("llm")

        # ----------------------------------------------------
        # Compile
        # ----------------------------------------------------

        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]
        )

        # ----------------------------------------------------
        # Save tools
        # ----------------------------------------------------

        self.tools = {
            t.name: t
            for t in tools
        }

        # ----------------------------------------------------
        # Bind tools to Groq
        # ----------------------------------------------------

        self.model = model.bind_tools(tools)


    # ========================================================
    # LLM node
    # ========================================================

    def call_model(
        self,
        state: AgentState
    ):

        messages = state["messages"]

        if self.system:

            messages = [
                SystemMessage(
                    content=self.system
                )
            ] + messages

        message = self.model.invoke(messages)

        return {
            "messages": [message]
        }


    # ========================================================
    # Check tool call
    # ========================================================

    def exists_action(
        self,
        state: AgentState
    ):

        last_message = state["messages"][-1]

        return len(last_message.tool_calls) > 0


    # ========================================================
    # Execute tools
    # ========================================================

    def take_action(
        self,
        state: AgentState
    ):

        tool_calls = state["messages"][-1].tool_calls

        results = []

        for tool_call in tool_calls:

            print("\n" + "=" * 60)
            print("CALLING TOOL")
            print("=" * 60)

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print("Tool:", tool_name)
            print("Arguments:", tool_args)

            # ------------------------------------------------
            # Check tool exists
            # ------------------------------------------------

            if tool_name not in self.tools:

                results.append(
                    ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=f"Unknown tool: {tool_name}"
                    )
                )

                continue

            # ------------------------------------------------
            # Execute tool
            # ------------------------------------------------

            tool_result = self.tools[
                tool_name
            ].invoke(tool_args)

            # ------------------------------------------------
            # Return result to LLM
            # ------------------------------------------------

            results.append(
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=str(tool_result)
                )
            )

        print("\nBack to the model!")

        return {
            "messages": results
        }


# ============================================================
# 7. System prompt
# ============================================================

prompt = """
You are a smart research assistant.

Use the Tavily search tool whenever the user asks
for current, recent, or web-based information.

You may make multiple searches when necessary.

Do not search unnecessarily.

For weather questions, search for the current weather
instead of guessing.

After receiving search results, provide a concise
and useful final answer.
"""


# ============================================================
# 8. Groq model
# ============================================================

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ============================================================
# 9. Create agent
# ============================================================

abot = Agent(
    model=model,
    tools=[tool],
    system=prompt,
    checkpointer=memory
)


# ============================================================
# 10. QUESTION 1
# ============================================================

print("\n")
print("=" * 70)
print("QUESTION 1")
print("=" * 70)

messages = [
    HumanMessage(
        content="What is the current weather in San Francisco?"
    )
]

thread = {
    "configurable": {
        "thread_id": "1"
    }
}


# ============================================================
# 11. First execution
# ============================================================

print("\nRunning graph...")

for event in abot.graph.stream(
    {
        "messages": messages
    },
    thread
):

    print("\nEVENT:")

    for key, value in event.items():

        print("NODE:", key)
        print(value)


# ============================================================
# 12. Current state
# ============================================================

print("\n")
print("=" * 70)
print("CURRENT STATE")
print("=" * 70)

current_state = abot.graph.get_state(thread)

print(current_state)

print("\nNEXT NODE:")
print(current_state.next)


# ============================================================
# 13. HUMAN APPROVAL
# ============================================================

print("\n")
print("=" * 70)
print("HUMAN APPROVAL")
print("=" * 70)

print(
    "\nThe graph is paused before executing Tavily."
)

approval = input(
    "\nAllow Tavily search? (yes/no): "
).strip().lower()


# Accept both "yes" and "y"
if approval in ["yes", "y"]:

    print("\nHuman approved the tool call.")

    for event in abot.graph.stream(
        None,
        thread
    ):

        for key, value in event.items():

            print("\nNODE:", key)
            print(value)

else:

    print("\nTool execution rejected by human.")


# ============================================================
# 14. QUESTION 2
# ============================================================

print("\n")
print("=" * 70)
print("QUESTION 2")
print("=" * 70)

messages = [
    HumanMessage(
        content="What is the current weather in Los Angeles?"
    )
]

thread2 = {
    "configurable": {
        "thread_id": "2"
    }
}


# ============================================================
# 15. First execution for Question 2
# ============================================================

for event in abot.graph.stream(
    {
        "messages": messages
    },
    thread2
):

    for key, value in event.items():

        print("\nNODE:", key)
        print(value)


# ============================================================
# 16. Inspect state
# ============================================================

print("\n")
print("=" * 70)
print("QUESTION 2 STATE")
print("=" * 70)

current_state = abot.graph.get_state(thread2)

print(current_state)

print("\nNEXT:")
print(current_state.next)


# ============================================================
# 17. Inspect messages
# ============================================================

print("\n")
print("=" * 70)
print("MESSAGES")
print("=" * 70)

for message in current_state.values["messages"]:

    print("\nMESSAGE:")
    print(message)


# ============================================================
# 18. Inspect tool calls
# ============================================================

last_message = current_state.values["messages"][-1]

print("\n")
print("=" * 70)
print("TOOL CALLS")
print("=" * 70)

print(last_message.tool_calls)


# ============================================================
# 19. HUMAN MODIFIES TOOL CALL
# ============================================================

if last_message.tool_calls:

    original_tool_call = last_message.tool_calls[0]

    tool_call_id = original_tool_call["id"]

    print("\nOriginal tool call:")
    print(original_tool_call)


    # --------------------------------------------------------
    # Human modifies the search query
    # --------------------------------------------------------

    modified_tool_call = {
        "name": original_tool_call["name"],
        "args": {
            "query": "current weather in Los Angeles California",
            "search_depth": "basic"
        },
        "id": tool_call_id
    }


    last_message.tool_calls = [
        modified_tool_call
    ]


    # --------------------------------------------------------
    # Update state
    # --------------------------------------------------------

    abot.graph.update_state(
        thread2,
        {
            "messages": [last_message]
        }
    )


    print("\nModified tool call:")

    updated_state = abot.graph.get_state(
        thread2
    )

    print(
        updated_state.values["messages"][-1].tool_calls
    )


# ============================================================
# 20. Execute modified tool
# ============================================================

print("\n")
print("=" * 70)
print("EXECUTING MODIFIED TOOL CALL")
print("=" * 70)

for event in abot.graph.stream(
    None,
    thread2
):

    for key, value in event.items():

        print("\nNODE:", key)
        print(value)



