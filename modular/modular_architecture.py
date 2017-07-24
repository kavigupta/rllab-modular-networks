
from component import FCNetwork, ImageNetwork
from tensor_cloud import input_tensor, TensorCloud

from functools import lru_cache

COLORS = "red", "green", "yellow", "blue"
DIMENSIONS = 3

def construct_network(hidden_size, number_layers, conv_size, n_conv_layers, image_width, image_height, filter_size=4):
    hidden_layers = [hidden_size] * number_layers
    state_to_features = FCNetwork("state_to_features", len(COLORS) * DIMENSIONS, hidden_layers, hidden_size)
    image_to_features = ImageNetwork("image_to_features",
                                    filter_size=filter_size,
                                    in_width=image_width, in_height=image_height,
                                    channel_sizes=[conv_size] * n_conv_layers, output_channels=conv_size, stride=1,
                                    hidden_layers=hidden_layers, hidden_size=hidden_size)
    features_to_color = {
        color : FCNetwork(f"features_to_{color}", hidden_size, hidden_layers, hidden_size)
            for color in COLORS
    }
    reacher_to_protocol = FCNetwork("reacher_to_protocol", hidden_size, hidden_layers, hidden_size)
    pusher_to_protocol = FCNetwork("blockpush_to_protocol", 2 * hidden_size, hidden_layers, hidden_size)
    protocol_to_linker = lru_cache(None)(lambda links: FCNetwork(
            f"protocol_to_{links}linker", hidden_size + 2 * DIMENSIONS * links, hidden_layers, DIMENSIONS * links)
    )
    state = TensorCloud.input(len(COLORS) * DIMENSIONS, "block_locations")
    joint_angles = TensorCloud({links : input_tensor(2 * DIMENSIONS * links, "joint_angles_%s" % links) for links in (3, 4, 5)})
    images = TensorCloud.input([image_width, image_height, 3], "images")
    images_features = images | image_to_features
    state_features = state | state_to_features
    features = state_features.label("state") + images_features.label("images")
    color_locs = features | features_to_color
    reach_protocol = color_locs | reacher_to_protocol
    push_protocol = TensorCloud.product(color_locs, color_locs, "concat_push", lambda x, y: x < y) | pusher_to_protocol
    robot_input = TensorCloud.product(push_protocol + reach_protocol, joint_angles, "concat_protocol_angles")
    output = robot_input | (lambda x: protocol_to_linker(x[-1]))
    return (images, state, joint_angles), output
