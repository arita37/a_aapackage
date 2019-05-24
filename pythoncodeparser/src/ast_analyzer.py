import ast


def analyzeSource(source):
    """Analyzes the given Python source

  Args:
    source: string

  Returns:
    A dict over all variables encountered in the tree, in following form:
    {(ast.stmt function_or_class,
      str variable_name): bool is_local, ...}
  """
    tree = ast.parse(source)
    analyzer = ASTAnalyzer()
    analyzer.visit(tree)

    return analyzer.getVariables()


class ASTAnalyzer(ast.NodeVisitor):
    """Visits nodes in a Python AST and collects information on variables"""
    LOCAL = "local"
    GLOBAL = "global"
    FORCE_GLOBAL = "force_global"

    def __init__(self):
        self.function_or_class = None
        self.variables = {}

    def getVariables(self):
        """Lists variables parsed from the given AST

    Returns:
      A dict over all variables encountered in the tree, in following form:
      {(ast.stmt function_or_class,
        str variable): bool is_local, ...}
    """
        label_to_bool = {self.__class__.LOCAL: True, self.__class__.GLOBAL: False, self.__class__.FORCE_GLOBAL: False}

        return {k: label_to_bool[v] for (k, v) in self.variables.items()}

    def _handleArguments(self, arguments):
        for arg in arguments.args:
            self._handleLocalVariable(variable_name=arg.arg)

    def _handleForceGlobalVariable(self, variable_name):
        key = (self.function_or_class, variable_name)

        self.variables[key] = self.__class__.FORCE_GLOBAL

    def _handleGlobalVariable(self, variable_name):
        key = (self.function_or_class, variable_name)

        self.variables.setdefault(key, self.__class__.GLOBAL)

    def _handleLocalVariable(self, variable_name):
        key = (self.function_or_class, variable_name)

        if self.variables.get(key) != self.__class__.FORCE_GLOBAL:
            self.variables[key] = self.__class__.LOCAL

    def _handleVariable(self, node):
        is_local = False
        if self.function_or_class is not None:
            if type(node.ctx) in [ast.Param, ast.Store, ast.AugStore]:
                is_local = True

        variable_name = node.id

        if is_local:
            self._handleLocalVariable(variable_name)
        else:
            self._handleGlobalVariable(variable_name)

        self.generic_visit(node)

    def _impl_visit_Function(self, node):
        old, self.function_or_class = self.function_or_class, node

        self._handleArguments(node.args)

        self.generic_visit(node)
        self.function_or_class = old

    def visit_AsyncFunctionDef(self, node):
        self._impl_visit_Function(node)

    def visit_ClassDef(self, node):
        old, self.function_or_class = self.function_or_class, node
        self.generic_visit(node)
        self.function_or_class = old

    def visit_FunctionDef(self, node):
        self._impl_visit_Function(node)

    def visit_Global(self, node):
        for variable_name in node.names:
            self._handleForceGlobalVariable(variable_name)

        self.generic_visit(node)

    def visit_Lambda(self, node):
        self._impl_visit_Function(node)

    def visit_Name(self, node):
        self._handleVariable(node)
