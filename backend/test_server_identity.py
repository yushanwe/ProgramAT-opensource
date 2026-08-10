import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


def load_resolver(socket_module):
    source_path = Path(__file__).resolve().parent / "stream_server.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_resolve_server_identity"
    )
    namespace = {"socket": socket_module}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_resolve_server_identity"]


class ServerIdentityTests(unittest.TestCase):
    def test_returns_resolved_hostname_and_ip(self):
        resolver = load_resolver(SimpleNamespace(
            gethostname=lambda: "programat-server",
            gethostbyname=lambda _hostname: "192.0.2.10",
        ))

        self.assertEqual(("programat-server", "192.0.2.10"), resolver())

    def test_unresolvable_hostname_is_non_fatal(self):
        def fail_resolution(_hostname):
            raise OSError("hostname not found")

        resolver = load_resolver(SimpleNamespace(
            gethostname=lambda: "local-hostname",
            gethostbyname=fail_resolution,
        ))

        self.assertEqual(("local-hostname", "unknown"), resolver())


if __name__ == "__main__":
    unittest.main()
