
import sys, ast, traceback, io
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
src = open(sys.argv[1]).read()
lines = src.splitlines()
FAIL, OK = [], [0]
def _soft(cond, line, msg=None):
    try:
        good = bool(cond)
    except Exception as e:
        good = False; msg = f"{msg} ({e})"
    if good:
        OK[0] += 1
    else:
        FAIL.append(f"ASSERT FAILED line {line}: {lines[line-1].strip()[:200]}" + (f"  msg={msg!r}" if msg is not None else ""))
class T(ast.NodeTransformer):
    def visit_Assert(self, node):
        call = ast.Call(func=ast.Name("_soft", ast.Load()),
                        args=[node.test, ast.Constant(node.lineno)] + ([node.msg] if node.msg else []), keywords=[])
        return ast.copy_location(ast.Expr(call), node)
try:
    tree = T().visit(ast.parse(src))
except SyntaxError as e:
    print(f"SYNTAX ERROR line {e.lineno}: {e.text!r} ({e.msg})"); sys.exit(2)
g = {"__name__": "__main__", "_soft": _soft}
for stmt in tree.body:
    mod = ast.fix_missing_locations(ast.Module(body=[stmt], type_ignores=[]))
    try:
        exec(compile(mod, sys.argv[1], "exec"), g)
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        ln = stmt.lineno
        FAIL.append(f"EXCEPTION in statement at line {ln}: {type(e).__name__}: {str(e)[:200]} | {lines[ln-1].strip()[:150]}")
print(f"passed assertions: {OK[0]}; failures: {len(FAIL)}")
for f in FAIL: print(f)
sys.exit(1 if FAIL else 0)
