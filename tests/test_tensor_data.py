import pytest
from hypothesis import given
from hypothesis.strategies import DataObject, data

import minitorch
from minitorch import TensorData

from .tensor_strategies import indices, tensor_data
from minitorch.tensor_data import to_index

# ## Tasks 2.1

# Check basic properties of layout and strides.


@pytest.mark.task2_1
def test_layout() -> None:
    "Test basis properties of layout and strides"
    data = [0] * 3 * 5
    tensor_data = minitorch.TensorData(data, (3, 5), (5, 1))

    assert tensor_data.is_contiguous()
    assert tensor_data.shape == (3, 5)
    assert tensor_data.index((1, 0)) == 5
    assert tensor_data.index((1, 2)) == 7

    tensor_data = minitorch.TensorData(data, (5, 3), (1, 5))
    assert tensor_data.shape == (5, 3)
    assert not tensor_data.is_contiguous()

    data = [0] * 4 * 2 * 2
    tensor_data = minitorch.TensorData(data, (4, 2, 2))
    assert tensor_data.strides == (4, 2, 1)


@pytest.mark.xfail
def test_layout_bad() -> None:
    "Test basis properties of layout and strides"
    data = [0] * 3 * 5
    minitorch.TensorData(data, (3, 5), (6,))


@pytest.mark.task2_1
@given(tensor_data())
def test_enumeration(tensor_data: TensorData) -> None:
    "Test enumeration of tensor_datas."
    indices = list(tensor_data.indices())

    # Check that enough positions are enumerated.
    assert len(indices) == tensor_data.size

    # Check that enough positions are enumerated only once.
    assert len(set(tensor_data.indices())) == len(indices)

    # Check that all indices are within the shape.
    for ind in tensor_data.indices():
        for i, p in enumerate(ind):
            assert p >= 0 and p < tensor_data.shape[i]


@pytest.mark.task2_1
@given(tensor_data())
def test_index(tensor_data: TensorData) -> None:
    "Test enumeration of tensor_data."
    # Check that all indices are within the size.
    for ind in tensor_data.indices():
        pos = tensor_data.index(ind)
        assert pos >= 0 and pos < tensor_data.size

    base = [0] * tensor_data.dims
    with pytest.raises(minitorch.IndexingError):
        base[0] = -1
        tensor_data.index(tuple(base))

    if tensor_data.dims > 1:
        with pytest.raises(minitorch.IndexingError):
            base = [0] * (tensor_data.dims - 1)
            tensor_data.index(tuple(base))


@pytest.mark.task2_1
@given(data())
def test_permute(data: DataObject) -> None:
    td = data.draw(tensor_data())
    ind = data.draw(indices(td))
    td_rev = td.permute(*list(reversed(range(td.dims))))
    assert td.index(ind) == td_rev.index(tuple(reversed(ind)))

    td2 = td_rev.permute(*list(reversed(range(td_rev.dims))))
    assert td.index(ind) == td2.index(ind)

@pytest.mark.task2_1
def test_ordinal_out_of_range():
    shape = (2, 2)
    out_index = [0, 0]

    # If ordinal is out of range (size of tensor is 4 for 2x2), it should raise an error.
    with pytest.raises(ValueError):
        to_index(4, shape, out_index)

@pytest.mark.task2_1
def test_valid_conversion():
    shape = (2, 2)
    out_index = [0, 0]
    to_index(0, shape, out_index)
    assert out_index == [0, 0]

    to_index(1, shape, out_index)
    assert out_index == [0, 1]

    to_index(2, shape, out_index)
    assert out_index == [1, 0]

    to_index(3, shape, out_index)
    assert out_index == [1, 1]

@pytest.mark.task2_1
def test_enumeration_coverage():
    shape = (3, 3)
    all_indices = set()

    for i in range(9):  # as 3x3 = 9
        out_index = [0, 0]
        to_index(i, shape, out_index)
        all_indices.add(tuple(out_index))

    # Ensure that all positions from 0 to 8 inclusive produce unique indices
    assert len(all_indices) == 9


# ## Tasks 2.2

# Check basic properties of broadcasting.


@pytest.mark.task2_2
def test_shape_broadcast() -> None:
    c = minitorch.shape_broadcast((1,), (5, 5))
    assert c == (5, 5)

    c = minitorch.shape_broadcast((5, 5), (1,))
    assert c == (5, 5)

    c = minitorch.shape_broadcast((1, 5, 5), (5, 5))
    assert c == (1, 5, 5)

    c = minitorch.shape_broadcast((5, 1, 5, 1), (1, 5, 1, 5))
    assert c == (5, 5, 5, 5)

    with pytest.raises(minitorch.IndexingError):
        c = minitorch.shape_broadcast((5, 7, 5, 1), (1, 5, 1, 5))
        print(c)

    with pytest.raises(minitorch.IndexingError):
        c = minitorch.shape_broadcast((5, 2), (5,))
        print(c)

    c = minitorch.shape_broadcast((2, 5), (5,))
    assert c == (2, 5)

    c = minitorch.shape_broadcast((2, 5), (1, 5))
    assert c == (2, 5)

    c = minitorch.shape_broadcast((2, 5), (2, 1))
    assert c == (2, 5)

@pytest.mark.task2_2
def test_broadcast_index():
    c = [0]
    minitorch.broadcast_index((2, 3), (5, 5), (1,), c)
    assert c == [0]

    c = [0]
    minitorch.broadcast_index((2, 3), (5, 5), (5,), c)
    assert c == [3]

    c = [0, 0]
    minitorch.broadcast_index((1, 2, 3), (1, 5, 5), (5, 5), c)
    assert c == [2, 3]

    c = [0, 0, 0, 0]
    minitorch.broadcast_index((4, 3, 2, 1), (5, 5, 4, 4), (5, 5, 1, 1), c)
    assert c == [4, 3, 0, 0]

    c = [0]
    minitorch.broadcast_index((3,), (5,), (1,), c)
    assert c == [0]

    c = [0, 0, 0]
    minitorch.broadcast_index((1, 2, 3), (2, 5, 5), (1, 5, 5), c)
    assert c == [0, 2, 3]

@given(tensor_data())
def test_string(tensor_data: TensorData) -> None:
    tensor_data.to_string()
