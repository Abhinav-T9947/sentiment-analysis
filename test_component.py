import os
import streamlit as st
import streamlit.components.v1 as components


st.title("Custom Component Test")

component_path = os.path.join(
    os.path.dirname(__file__),
    "autocomplete_component",
    "frontend"
)

st.write("Component path:")
st.code(component_path)

st.write("Frontend exists:")

st.write(
    os.path.exists(
        os.path.join(
            component_path,
            "index.html"
        )
    )
)


test_component = components.declare_component(
    "test_component",
    path=component_path
)


value = test_component(
    default="TEST"
)

st.write("Returned value:", value)

