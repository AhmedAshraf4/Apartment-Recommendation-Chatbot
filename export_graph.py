from app.graph.workflow import chat_graph

graph = chat_graph.get_graph()
mermaid = graph.draw_mermaid()

with open("langgraph_mermaid.txt", "w", encoding="utf-8") as file:
    file.write(mermaid)
