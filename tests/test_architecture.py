import ast
import unittest

from tests.support import ROOT


class ArchitectureTests(unittest.TestCase):
    def test_import_directions(self):
        allowed = {
            "core": {"core"},
            "smt": {"core", "smt"},
            "ops": {"core", "smt", "ops"},
            "relation": {"core", "smt", "relation"},
            "stage": {"core", "relation", "stage"},
            "reduction": {"core", "relation", "stage", "reduction"},
            "verification": {
                "core",
                "smt",
                "ops",
                "relation",
                "stage",
                "reduction",
                "verification",
            },
            "app": {
                "core",
                "smt",
                "ops",
                "relation",
                "stage",
                "reduction",
                "verification",
                "app",
            },
        }
        for p in (ROOT / "src").rglob("*.py"):
            parts = p.relative_to(ROOT).with_suffix("").parts
            if len(parts) < 3:
                continue
            package = list(parts[:-1])
            owner = parts[1]
            self.assertIn(owner, allowed, p)
            for node in ast.walk(ast.parse(p.read_text())):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [a.name.split(".") for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [
                        package[: len(package) - node.level + 1]
                        + (node.module or "").split(".")
                        if node.level
                        else (node.module or "").split(".")
                    ]
                for m in modules:
                    if len(m) > 1 and m[0] == "src":
                        self.assertIn(m[1], allowed[owner], f"{p}:{node.lineno}")
                    if owner == "core":
                        self.assertNotEqual(m[0], "z3", p)

    def test_no_import_time_registration(self):
        for p in (ROOT / "src").rglob("__init__.py"):
            tree = ast.parse(p.read_text())
            self.assertFalse(any(isinstance(n, ast.Call) for n in ast.walk(tree)), p)
