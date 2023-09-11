from copy import deepcopy

import pytest

from src.core.node import (
    Node,
    create_tree,
    get_max_height_nodes,
    health_check,
    reindex_tree,
    remove_from_tree,
)


@pytest.fixture
def root():
    height, num_classes = 4, 5
    tree_root = create_tree(height=height, num_classes=num_classes)
    return tree_root


class TestTree:
    def test_create_tree(self):
        height, num_classes = 4, 5
        tree = create_tree(height=height, num_classes=num_classes)
        health_check(tree)
        assert tree.max_height == height
        assert len(tree.descendants) == 2 ** (height + 1) - 1
        assert len(tree.leaves) == 2**height
        assert tree.index == 0
        assert tree.parent is None
        assert tree.left.parent == tree.right.parent == tree

    def test_reindex(self, root):
        reindexed_tree = deepcopy(root)
        reindex_tree(reindexed_tree)

        # root was already indexed, so the structure should be the same
        # can't use .descendants because they are a set and the order is not guaranteed
        cur_n_to_compare = [root], [reindexed_tree]
        for d in range(root.max_height):
            for node1, node2 in zip(*cur_n_to_compare):
                assert node1.index == node2.index
            cur_n_to_compare = (
                [child for node in cur_n_to_compare[0] for child in node.child_nodes],
                [child for node in cur_n_to_compare[1] for child in node.child_nodes],
            )

    def test_paths(self, root):
        sample_leaf = next(iter(root.leaves))
        leaf_path_from_root = sample_leaf.get_path_from_ancestor()
        assert len(leaf_path_from_root) == root.max_height + 1
        assert leaf_path_from_root[-1] == sample_leaf
        assert leaf_path_from_root[0] == root
        assert root.left.get_path_from_ancestor() == [root, root.left]

    def test_node_properties(self):
        mini_tree = create_tree(height=1, num_classes=2)
        assert mini_tree.is_root and not mini_tree.is_leaf
        assert not mini_tree.is_left_child and not mini_tree.is_right_child
        leaf1, leaf2 = mini_tree.left, mini_tree.right
        assert leaf1.is_leaf
        assert leaf2.is_leaf
        assert leaf1.is_left_child and not leaf1.is_right_child
        assert leaf2.is_right_child and not leaf2.is_left_child

    def test_get_max_height_nodes(self):
        height = 6
        root = create_tree(height=height, num_classes=2)

        def all_leave_idx_exceed_10(node: Node):
            threshold = 10
            for leaf in node.leaves:
                if leaf.index <= threshold:
                    return False
            return True

        max_height_nodes = get_max_height_nodes(root, all_leave_idx_exceed_10)
        # Have a look at root.print_tree() to understand why this is the correct result for a tree of height 6
        assert {node.index for node in max_height_nodes} == {11, 18, 33, 64}

    def test_remove_node(self, root):
        node_to_remove = root.left.left
        n_nodes_before = len(root.descendants)
        n_nodes_to_remove = len(node_to_remove.descendants) + 1
        remove_from_tree(node_to_remove)
        health_check(root)
        assert len(root.descendants) == n_nodes_before - n_nodes_to_remove
        assert node_to_remove.is_root

        # the removed tree should be a healthy tree in itself
        reindex_tree(node_to_remove)
        assert node_to_remove.index == 0
        health_check(node_to_remove)
