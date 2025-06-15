from pylint.checkers import BaseChecker


class RandomSeedChecker(BaseChecker):
    name = 'random-seed-checker'
    msgs = {
        'W9001': (
            'Hardcoded random seed detected; consider parameterizing it.',
            'hardcoded-random-seed',
            'Used when random seeds are hardcoded directly in the code.',
        ),
    }

    def visit_call(self, node):
        if getattr(node.func, 'attrname', '') == 'seed':
            if node.args and node.args[0].as_string().isdigit():
                self.add_message('hardcoded-random-seed', node=node)


def register(linter):
    linter.register_checker(RandomSeedChecker(linter))
