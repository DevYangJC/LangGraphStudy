"""
SendDemo.py
LangGraph Map-Reduce 模式演示
通过使用 Send 对象，LangGraph 提供了一种优雅的方式来实现这种动态图结构，
使得我们可以根据运行时状态来决定执行路径。

解释：
（1）首先执行 generate_subjects主题列表节点，生成主题列表：['猫', '狗', '程序员']
（2）然后通过条件边函数 map_subjects_to_jokes 为每个主题创建一个 Send 对象
（3）make_joke 节点被并行执行3次，每次处理一个主题
（4）最终将所有生成的笑话合并到一个列表中
这种模式非常适合处理动态数量的任务
"""

from typing import Annotated, List, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# 定义状态
class AtguiguState(TypedDict):
    subjects: List[str]
    jokes: Annotated[List[str], lambda x, y: x + y]  # 使用列表合并的方式


# 第一个节点：生成需要处理的主题列表
def generate_subjects(state: AtguiguState) -> dict:
    """生成需要处理的主题列表"""
    print("执行节点(第一个节点：生成需要处理的主题列表): generate_subjects")
    subjects = ["猫", "狗", "程序员"]
    print(f"生成主题列表: {subjects}")
    return {"subjects": subjects}


# Map节点：为每个主题生成笑话
def make_joke(state: AtguiguState) -> dict:
    """为单个主题生成笑话"""
    subject = state.get("subject", "未知")
    print(f"执行节点: make_joke，处理主题: {subject}")

    # 根据主题生成相应笑话
    jokes_map = {
        "猫": "为什么猫不喜欢在线购物？因为它们更喜欢实体店！",
        "狗": "为什么狗不喜欢计算机？因为它们害怕被鼠标咬！",
        "程序员": "为什么程序员喜欢洗衣服？因为他们在寻找bugs！",
        "未知": "这是一个关于未知主题的神秘笑话。"
    }

    joke = jokes_map.get(subject, f"这是一个关于{subject}的即兴笑话。")
    print(f"生成笑话: {joke}")
    return {"jokes": [joke]}


# 条件边函数：根据主题列表生成Send对象列表
def map_subjects_to_jokes(state: AtguiguState) -> List[Send]:
    """将主题列表映射到joke生成任务"""
    print("执行条件边函数: map_subjects_to_jokes")
    subjects = state["subjects"]
    print(f"映射主题到joke任务: {subjects}")

    # 为每个主题创建一个Send对象，指向make_joke节点
    # 每个Send对象包含节点名称和传递给该节点的状态
    send_list = [Send("make_joke", {"subject": subject}) for subject in subjects]
    print(f"生成Send对象列表: {send_list}")
    return send_list


def main():
    """演示Map-Reduce模式"""
    print("=== Map-Reduce 模式演示 ===\n")

    # 创建图
    builder = StateGraph(AtguiguState)

    # 添加节点
    builder.add_node("generate_subjects", generate_subjects)
    builder.add_node("make_joke", make_joke)

    # 添加边
    builder.add_edge(START, "generate_subjects")

    # 添加条件边，使用Send对象实现map-reduce
    builder.add_conditional_edges(
        "generate_subjects",  # 源节点
        map_subjects_to_jokes  # 路由函数，返回Send对象列表
    )

    # 从make_joke到结束
    builder.add_edge("make_joke", END)

    # 编译图
    graph = builder.compile()
    print(graph.get_graph().print_ascii())

    # 执行图
    initial_state = {"subjects": [], "jokes": []}
    print("初始状态:", initial_state)
    print("\n开始执行图...")

    result = graph.invoke(initial_state)
    print(f"\n最终结果: {result}")

    print("\n=== 演示完成 ===")


'''
map_subjects_to_jokes 是一个条件边路由函数，它返回一个 Send 对象列表。
在 LangGraph 中，当条件边返回多个 Send 时，每个 Send 都会独立地启动一个任务，指向指定的节点（这里都是 "make_joke"），
    并携带各自的状态数据（这里是 {"subject": "猫"}、{"subject": "狗"}、{"subject": "程序员"}）。

这三个任务会被并行执行（由 LangGraph 的运行时调度），每个 make_joke 节点实例都会根据传入的 subject 生成一个笑话，并返回一个包含 jokes 列表的状态。

最后，由于 jokes 字段使用了 reducer（Annotated[List[str], lambda x, y: x + y]），所有并行节点返回的 jokes 列表会被依次合并成一个总列表，从而完成 Map‑Reduce 中的“Reduce”步骤。

所以你的理解非常到位：路由函数产生三个 Send 对象 → 三个任务并行执行 → 结果汇总。这正是该示例的核心机制。👍
'''

if __name__ == "__main__":
    main()
