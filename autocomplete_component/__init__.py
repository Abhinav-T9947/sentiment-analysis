import os
import streamlit.components.v1 as components


_COMPONENT_PATH = os.path.join(
    os.path.dirname(__file__),
    "frontend"
)


_review_input = components.declare_component(
    "review_input_v2",
    path=_COMPONENT_PATH
)


def autocomplete_review(
    value="",
    suggestions=None,
    placeholder="Start typing your review..."
):

    return _review_input(
        value=value,
        suggestions=suggestions or [],
        placeholder=placeholder,
        default=value
    )

