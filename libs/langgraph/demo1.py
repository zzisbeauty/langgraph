from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END

# 1. 定义状态（这是所有 Agent 共享的黑板）
class AgentState(TypedDict):
    messages: Annotated[list, "对话历史"]
    next_step: str  # 下一个该谁干活？

def supervisor(state: AgentState):
    print("--- [主管] 正在思考下一步... ---")
    last_message = state["messages"][-1]
    if "资料" not in last_message:
        return {"next_step": "RESEARCHER"}
    elif "写好了" not in last_message:
        return {"next_step": "WRITER"}
    else:
        return {"next_step": "FINISH"}

def research_agent(state: AgentState):
    print("--- [研究员] 正在搜集资料... ---")
    return {"messages": state["messages"] + ["研究员：我找到了相关资料。"]}

def writer_agent(state: AgentState):
    print("--- [作家] 正在润色文字... ---")
    return {"messages": state["messages"] + ["作家：我已经根据资料写好了。"]}

# 4. 构建图 (The Graph)
builder = StateGraph(AgentState)

builder.add_node("RESEARCHER", research_agent)
builder.add_node("WRITER", writer_agent)
builder.add_node("SUPERVISOR", supervisor)

# 5. 设置连线（Edges）
builder.set_entry_point("SUPERVISOR") # 从主管开始

# 根据主管的判断跳转
builder.add_conditional_edges(
    "SUPERVISOR",
    lambda x: x["next_step"],
    {
        "RESEARCHER": "RESEARCHER",
        "WRITER": "WRITER",
        "FINISH": END
    }
)

# 工人干完活，总是回到主管那里
builder.add_edge("RESEARCHER", "SUPERVISOR")
builder.add_edge("WRITER", "SUPERVISOR")

graph = builder.compile()



# 6. 启动图并传入初始输入
initial_input = {
    "messages": ["请帮我写一篇关于人工智能未来发展的研究报告"],
    "next_step": "SUPERVISOR"
}

# 运行并打印结果
for event in graph.stream(initial_input):
    for node_name, output in event.items():
        print(f"--- 正在执行节点: {node_name} ---")
        print(output)
        print("-" * 30)