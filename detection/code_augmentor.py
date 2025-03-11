import builtins
import re
import ast

import astor


class CodeAugmentor:
    @staticmethod
    def remove_comments(code):
        code = re.sub(r'\s*""".*?"""', '', code, flags=re.DOTALL)

        edited_lines = []
        code_lines = code.splitlines()
        for line in code_lines:
            if line.strip().startswith('#'):
                continue
            edited_lines.append(line)
        code = '\n'.join(edited_lines)
        code = re.sub(r' #.*', '', code)

        return code


