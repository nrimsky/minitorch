from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
from collections import defaultdict

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_copy = list(vals)
    vals_copy[arg] += epsilon
    new_f = f(*vals_copy)
    return (new_f - f(*vals)) / epsilon

variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    stack = []
    def visit(node):
        if node.unique_id not in visited:
            visited.add(node.unique_id)
            if not node.is_constant():
                for parent in node.parents:
                    visit(parent)
            stack.append(node)
    visit(variable)
    return stack[::-1]

def backpropagate(variable: Variable, deriv: Any) -> None:
    derivative_dict = defaultdict(float)
    derivative_dict[variable.unique_id] = float(deriv)
    topo_order = topological_sort(variable)

    for node in topo_order:
        if not node.is_leaf():
            for parent, d_parent in node.chain_rule(derivative_dict[node.unique_id]):
                if parent.is_leaf():
                    parent.accumulate_derivative(d_parent)
                else:
                    derivative_dict[parent.unique_id] += d_parent


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
