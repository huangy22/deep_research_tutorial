import random
import secrets
from typing import ClassVar, List

from google.adk.agents import BaseAgent, ParallelAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event, EventActions
from google.genai import types

from ..llm_config import create_default_config
from .paper_agent.agent import init_paper_agent
from .report_agent.agent import init_report_agent


def paper_list_before_agent(callback_context: CallbackContext):
    if (
        callback_context.state.get('database_agent_tool_call', None) is None
        or len(callback_context.state['database_agent_tool_call']) == 0
    ):
        callback_context._event_actions.escalate = True
        return
    
    toolcall_infos = callback_context.state['database_agent_tool_call']
    while toolcall_infos[-1]['tool_name'] != 'query_table' or toolcall_infos[-1]['tool_response']['paper_count'] == 0:
        toolcall_infos.pop()
    query_table_info = toolcall_infos[-1]

    paper_list = query_table_info['tool_response'].get('papers', [])
    if paper_list is None or len(paper_list) == 0:
        callback_context._event_actions.escalate = True
        return
    # use 5 papers for demo
    callback_context.state['paper_list'] = {f'paper{i+1}': paper for i, paper in enumerate(paper_list[:5])}
    return

class GroupPaperAgent(BaseAgent):
    """Distributes tasks and dynamically creates a ParallelAgent."""

    def __init__(self, name):
        super().__init__(name=name)
        self.name = name

    async def _run_async_impl(self, ctx):
        paper_list = ctx.session.state['paper_list']
        POOL = list(paper_list.keys())
        run_id = secrets.token_hex(2)

        task_delta = {f"task:{run_id}:{name}": random.randint(1, 9)
                      for name in POOL}
        yield Event(
            author=self.name,
            content=types.Content(role=self.name,
                                  parts=[types.Part(text=f"running tasks for {len(POOL)} paper readings")]),
            actions=EventActions(state_delta={"current_run": run_id, **task_delta})
        )
        parallel = ParallelAgent(
            name=f"block_{run_id}",
            sub_agents=[init_paper_agent(llm_config, name=n, run_id=run_id) for n in POOL]
        )
        async for ev in parallel.run_async(ctx):
            yield ev


def init_paper_group_agent():
    return GroupPaperAgent(name="group_paper")


def init_deep_research_agent(llm_config):
    paper_group_agent = init_paper_group_agent()
    report_agent = init_report_agent(llm_config)

    root_agent = SequentialAgent(
        name="deep_research_agent",
        description="The agent to perform deep research literature review and generate a scientific report",
        sub_agents=[paper_group_agent, report_agent],
        before_agent_callback=paper_list_before_agent
    )
    return root_agent


# Example usage
llm_config = create_default_config()
root_agent = init_deep_research_agent(llm_config)
