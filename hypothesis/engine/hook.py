"""
Hypothesis `hook` sub-system.

The module describes the functionality of Hypothesis hooks. A hook is a
location in the user's and framework's code in which arbitrary code can
be injected. A hook can take on additional arguments depending on
its definition.

This can be useful in situations where one would like to keep track of some
internal variable, or for logging purposes. For instance, when performing
variational inference, one would like to keep track of the negative
log likelihood of the proposal distribution with respect to the true
solution for plotting purposes.

When extending Hypothesis with additional algorithms, hooks can be
added in the constructor of the class, and will remain in Hypothesis
during runtime unless these are cleared.
"""

import numpy as np



"""Main hooks storage."""
hooks = {}


"""Main tag storage."""
class Tags:
    pass
# Allocate the data storage.
tags = Tags()


def add_tag(tag):
    r"""Adds the specified hook-tag to Hypothesis at runtime.

    Args:
        tag (str): the name of the hook to allocate.
    """
    # Check if the tag already exists.
    if hasattr(tags, tag):
        raise ValueError("Tag " + tag + " already exists.")
    setattr(tags, tag, np.random.randint(-2 ** 63, 2 ** 63))


def remove_tag(tag):
    r"""Removes the specified hook-tag from Hypothesis.

    Args:
        tag (str): the name of the hook to remove.
    """
    if hasattr(tags, tag):
        delattr(tags, tag)


def call_hook(tag, argument, **kwargs):
    r"""Calls the hooks assigned to the specified hook-tag.

    Args:
        tag (str): the hook-identifier to call.
        argument (object):
    """
    if tag in hooks.keys() and len(hooks[tag]) > 0:
        for f in hooks[tag]:
            f(argument, **kwargs)


def register_hook(tag, f):
    r"""Registers the lambda function `f` under the specified hook-tag.

    Args:
        tag (str): the hook-tag to attach to.
        f (lambda): the lambda function to run when the hook is called.
    """
    # Check if the tag is present in the hooks storage.
    if tag not in hooks.keys():
        hooks[tag] = []
    else:
        raise ValueError("Specified hook-tag does not exist.")
    # Add the call-back.
    if f not in hooks[tag]:
        hooks[tag].append(f)


def clear_hook(tag=None):
    r"""Removes the hook-tag from the tag-list.

    If no hook-tag has been specified, all hook-tags will be removed.

    Args:
        tag (str, optional): the hook-tag to remove.
    """
    # Check if a tag has been specified.
    if tag:
        del hooks[tag]
    # Clear all registered hooks.
    else:
        keys = list(hooks.keys())
        for key in keys:
            del hooks[key]


"""Add default Hypothesis hook-tags."""
add_tag("start")
add_tag("end")
