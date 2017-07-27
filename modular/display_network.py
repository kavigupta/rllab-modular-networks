
import tensorflow as tf
import numpy as np
#from https://stackoverflow.com/questions/38189119/simple-way-to-visualize-a-tensorflow-graph-in-jupyter
from IPython.display import clear_output, Image, display, HTML

def show_graph(path=None):
    """Visualize TensorFlow graph."""
    graph_def = tf.get_default_graph().as_graph_def()
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(graph_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    result = iframe
    if path is None:
        display(HTML(result))
    else:
        with open(path, "w") as f:
            f.write(result)
