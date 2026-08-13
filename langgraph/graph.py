"""Tiny langgraph.graph stub implementing just enough API used by app.py
This is a minimal in-repo replacement so tests and deterministic runs work
without installing the external `langgraph` package. It is intentionally
small and synchronous.
"""
from typing import Callable, Dict, Any

END = object()


class StateGraph:
    def __init__(self, state_type=None):
        self.nodes = {}
        self.entry = None
        self.edges = {}
        self.cond = {}

    def add_node(self, name: str, fn: Callable[[Dict], Dict]):
        self.nodes[name] = fn

    def set_entry_point(self, name: str):
        self.entry = name

    def add_edge(self, frm: str, to: str):
        self.edges.setdefault(frm, []).append(to)

    def add_conditional_edges(self, node: str, cond_fn: Callable[[Dict], str], mapping: Dict[str, str]):
        self.cond[node] = (cond_fn, mapping)

    def compile(self):
        nodes = self.nodes
        edges = self.edges
        cond = self.cond
        entry = self.entry

        class Runner:
            def __init__(self):
                pass

            def invoke(self, initial_state: Dict[str, Any]):
                state = initial_state
                cur = entry
                # run entry node
                while True:
                    if cur is None:
                        break
                    fn = nodes.get(cur)
                    if not fn:
                        break
                    state = fn(state)
                    # conditional edge handling
                    if cur in cond:
                        cond_fn, mapping = cond[cur]
                        nxt = cond_fn(state)
                        cur = mapping.get(nxt, None)
                        if cur is None:
                            break
                        continue
                    # normal single outgoing edge
                    outs = edges.get(cur, [])
                    if not outs:
                        break
                    cur = outs[0]
                return state

        return Runner()
