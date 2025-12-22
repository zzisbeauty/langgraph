import getpass
import os

# 这段代码的作用是安全地配置你的环境变量，特别是为了保护你的 **API 密钥（Key）**不被泄露。

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

# _set_if_undefined("vllm_key")
# _set_if_undefined("langsmith")

# _set_if_undefined("ANTHROPIC_API_KEY")
_set_if_undefined("TAVILY_API_KEY")
# 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





# utils - llm


from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# openai 兼容 API
def vllmmodelserver(base_url="http://vllm-server-base-builddev-image:8000/v1",model_name="/localmodels/Qwen3-4B-Thinking-2507", api_key=""):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name, timeout=32768)
    return llm

# import sys, os

# print(f"当前使用的 Python 路径: {sys.executable}")
# print(f"当前包搜索路径: {sys.path}")

# try:
#     import socksio
#     print("socksio 已成功安装且可访问")
# except ImportError:
#     print("错误：socksio 仍然无法被当前环境识别")
    
llm = vllmmodelserver()








# 定义工具
from typing import Annotated
from langchain_tavily import TavilySearch
from langchain_core.tools import tool


from langchain_experimental.utilities import PythonREPL

repl = PythonREPL()


#  =========#  =========#  =========#  =========#  ========= 创建 TOOLS


""" “多 Agent 协作”（Multi-Agent Collaboration）架构

# 它实现了一个双 Agent 循环系统：一个负责搜索（Researcher），一个负责绘图（Chart Generator），两者交替工作直到得出最终答案。
"""

# tools -1 
tavily_tool = TavilySearch(max_results=5)


# tools - 2

@tool
def python_repl_tool(code: Annotated[str, "用于生成你的图表并执行的 Python 代码"],):
    """使用此工具来执行 Python 代码。如果你想查看某个值的输出，应当使用 print(...) 将其打印出来。这样用户就可以看到结果。"""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER.")


# @tool  
# def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."]):  
#     """Use this to execute python code. If you want to see the output of a value,  
#     you should print it out with `print(...)`. This is visible to the user."""  
#     try:  
#         # 添加 matplotlib 配置  
#         import matplotlib  
#         matplotlib.use('Agg')  # 使用非交互式后端  
#         import matplotlib.pyplot as plt  
          
#         result = repl.run(code)  
          
#         # 如果代码中创建了图表，保存并显示  
#         if 'plt' in code or 'matplotlib' in code:  
#             plt.savefig('/tmp/chart.png', dpi=150, bbox_inches='tight')  
#             plt.close()  
#             return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}\n\nChart saved to /tmp/chart.png"  
          
#     except BaseException as e:  
#         return f"Failed to execute. Error: {repr(e)}"  
      
#     return (  
#         result + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."  
#     )


# # 单独测试 PythonREPL 图表生成  
# test_code = """  
# import matplotlib.pyplot as plt  
# years = [2019, 2020, 2021, 2022, 2023]  
# gdp = [2.83, 2.76, 3.13, 3.07, 3.13]  
  
# plt.figure(figsize=(10, 6))  
# plt.plot(years, gdp, marker='o')  
# plt.title('UK GDP (2019-2023)')  
# plt.savefig('test_chart.png')  
# print("Chart saved as test_chart.png")  
# """  
# result = python_repl_tool.invoke(test_code)  
# print(result)



# ==================# ==================# ==================# ==================# ==================

from typing import Literal
from langgraph.types import Command
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END




# ========# ========# ========# ========# ========# ========# ========  创建  Agent  前的准备



# Agent 调用得到其需要的系统提示 -  协作协议 - 系统级提示
def make_system_prompt(suffix: str) -> str:
    return (
        """
        你是一名有帮助的 AI 助手，与其他助手协作。
        使用所提供的工具来推进问题的解答。
        如果你无法完全回答，也没关系，另一位拥有不同工具的助手
        会在你停下的地方继续帮助。请尽你所能执行并取得进展。
        """
        f"\n{suffix}"
    )

# 如果你或其他任何助手已经得出最终答案或可交付成果，
#         请在你的回复前加上 FINAL ANSWER，以便团队知道可以停止。


# ========# ========# ========# ========# ========# ========# ========  创建  Agent


# Research agent and node =========== Agent 1
# 方法解释 https://share.google/aimode/C942mfmtaA5H8atE3
# 这个 Agent 和 node 之间是什么关系。它里面有 node 这个概念代表的功能模块？  https://share.google/aimode/Ec9lXcc46uAEzJzlH
research_agent = create_react_agent(
    llm,
    tools=[tavily_tool], # 绑定工具
    prompt=make_system_prompt(
        "You can only do research. You are working with a chart generator colleague. "  
        "IMPORTANT: Never include 'FINAL ANSWER' in your response. "  
        "Always let the chart generator create the visualization."  
    ), # 节点级提示（Agent 专用 prompt）
)


# Chart generator agent and node =========== Agent 2
# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
# Agent 解释说明，主要是 tool 的绑定有疑问： https://share.google/aimode/9eXTN36FySagtEYVm
# 这个Agent的方法的详细调用过程： https://share.google/aimode/YtjLaSoDRBhWBrFH8   很重要，很清楚的说明  Tool 对于 Agent 的概念
chart_agent = create_react_agent(
    llm,
    [python_repl_tool],
    prompt=make_system_prompt("你负责生成图表。你正在与一位负责研究的同事协作。"),
)


# 上述两个 Agent 的 tool 的区别： https://share.google/aimode/XMpjXwINTFo0o5DKg








# 用于实现研究智能体和图表生成智能体之间的协作

# 路由逻辑函数
def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return goto


def research_node(state: MessagesState,) -> Command[Literal["chart_generator", END]]:
    result = research_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "chart_generator")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="researcher"
    )
    return Command(  # 返回 Command 对象，包含状态更新和路由信息
        update={
            # share internal message history of research agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )


def chart_node(state: MessagesState) -> Command[Literal["researcher", END]]:
    result = chart_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "researcher")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_generator"
    )
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )








# 定义图结构，并未执行

from langgraph.graph  import StateGraph,  START

workflow = StateGraph(MessagesState)
workflow.add_node("researcher", research_node)
workflow.add_node("chart_generator", chart_node)

# START 是一个特殊的虚拟节点，用于定义图的入口点。当你添加从 START 到某个节点的边时，你是在告诉图执行引擎："当图开始执行时，首先运行这个节点"。
workflow.add_edge(START, "researcher") #  图结构从这里开始
# graph = workflow.compile() # # 编译图，但还没有执行
graph = workflow.compile(  
    interrupt_before=["chart_generator"],  # 调试断点  
    interrupt_after=["researcher"]         # 调试断点  
)
# 编译图后，图结构中 Agent 将会根据 State 决定走向哪个 node 节点：
# https://deepwiki.com/search/-def-researchnodestate-message_8f0ae874-d6d3-49d2-a31a-d632ece50063?mode=fast#10








# 图结构的执行过程：
# 1. stream 和 invoke 执行图结构的过程：https://deepwiki.com/search/-def-researchnodestate-message_8f0ae874-d6d3-49d2-a31a-d632ece50063?mode=fast#11
# 2. ~~整个图结构定义的 Agent 的 prompts 依赖：https://deepwiki.com/search/-def-researchnodestate-message_8f0ae874-d6d3-49d2-a31a-d632ece50063?mode=fast#12~~
# 3. 图结构的执行过程： 在用户信息输入后，图结构中的 Agent 才会开始执行 https://deepwiki.com/search/-def-researchnodestate-message_8f0ae874-d6d3-49d2-a31a-d632ece50063?mode=fast#13

# 这段代码使用 graph.stream() 方法来流式执行图并实时输出每个节点的执行结果，而不是等待整个图执行完成。
# 向图提供用户任务：获取英国过去5年的GDP数据并制作图表。


# 需要提供 config 参数来支持断点  
config = {"configurable": {"thread_id": "debug-thread"}} 

events = graph.stream(
    {
        "messages": [ # 用户级输入（messages）；  当 Agent 执行时，这些指令会组合成完整的提示： [系统提示] + [节点提示] + [用户输入] → 完整的 LLM 提示  
            (
                "user",
                "First, get the UK's GDP over the past 5 years, then make a line chart of it. "
                "Once you make the chart, 务必保证图表被正确绘制。除了必要的英文信息，其他说明用中文输出。finish.",
            )
        ],
    },
    config,
    {"recursion_limit": 150}, # Maximum number of steps to take in the graph
)

from pprint import pprint

for event in events:
    pprint(event)  
    state = graph.get_state(config)
    print("当前状态值:", state.values)
    print("下一个节点:", state.next)
    print("最后消息:", state.values["messages"][-1].content[:200] + "...")
    pprint("----")



# # 查看所有事件的完整结构  
# for s in events:  
#     print("Event:", s)  
#     print("Event type:", type(s))  
#     print("Keys:", s.keys() if isinstance(s, dict) else "Not a dict")  
#     print("----")


# print('========================')


# # 记录执行的所有节点  
# executed_nodes = []  
# for s in events:  
#     if isinstance(s, dict):  
#         for node_name in s.keys():  
#             executed_nodes.append(node_name)  
  
# print("执行的节点顺序:", executed_nodes)

# # for s in events:
# #     print(s)
# #     print("----")

# print('========================')

# # # 在流式输出中查找 chart_generator 的响应  
# # for s in events:  
# #     if "chart_generator" in s:  
# #         print("Chart Generator Output:", s["chart_generator"]["messages"][-1].content)

