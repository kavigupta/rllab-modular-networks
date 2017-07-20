
from component import NetworkComponent
from tensor_cloud import input_tensor, TensorCloud

from functools import lru_cache

COLORS = "red", "green", "yellow", "blue"
DIMENSIONS = 3

def construct_network(hidden_size, number_layers):
    hidden_layers = [hidden_size] * number_layers
    state_to_features = NetworkComponent("state_to_features", len(COLORS) * DIMENSIONS, hidden_layers, hidden_size)
    features_to_color = {
        color : NetworkComponent(f"features_to_{color}", hidden_size, hidden_layers, hidden_size)
            for color in COLORS
    }
    reacher_to_protocol = NetworkComponent("reacher_to_protocol", hidden_size, hidden_layers, hidden_size)
    pusher_to_protocol = NetworkComponent("blockpush_to_protocol", 2 * hidden_size, hidden_layers, hidden_size)
    protocol_to_linker = lru_cache(None)(lambda links: NetworkComponent(
            f"protocol_to_{links}linker", hidden_size + 2 * DIMENSIONS * links, hidden_layers, DIMENSIONS * links)
    )
    state = TensorCloud.input(len(COLORS) * DIMENSIONS, "block_locations")
    joint_angles = TensorCloud({links : input_tensor(2 * DIMENSIONS * links, "joint_angles_%s" % links) for links in (3, 4, 5)})
    color_locs = state | state_to_features | features_to_color
    reach_protocol = color_locs | reacher_to_protocol
    push_protocol = TensorCloud.product(color_locs, color_locs, "concat_push", lambda x, y: x < y) | pusher_to_protocol
    robot_input = TensorCloud.product(push_protocol + reach_protocol, joint_angles, "concat_protocol_angles")
    output = robot_input | (lambda x: protocol_to_linker(x[-1]))
    return state, joint_angles, output
