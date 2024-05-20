# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Helpers for handling errors in user-provided functions."""

import functools
import logging
from typing import Any, Callable


def ignoring_arithmetic_errors(fun: Callable[..., Any]) -> Callable[..., Any]:
  """Makes fun return None instead of raising ArithmeticError."""

  @functools.wraps(fun)
  def fun_wrapped(*args):
    try:
      return fun(*args)
    except ArithmeticError:
      logging.warning(
          "Encountered arithmetic error in function: for value %s. "
          "Assuming this input will never occur.", str(args))
      return None

  return fun_wrapped
