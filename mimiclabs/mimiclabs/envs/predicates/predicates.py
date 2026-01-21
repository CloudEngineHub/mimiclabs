from ...utils import disable_module_import

with disable_module_import("libero", "libero", "envs"):
    from libero.libero.envs.predicates.base_predicates import *


class Grasp(UnaryAtomic):
    def __call__(self, arg1):
        return arg1.check_grasp()

class Still(UnaryAtomic):
    def __call__(self, arg1):
        return arg1.check_still()