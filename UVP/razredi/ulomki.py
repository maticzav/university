# =============================================================================
# Ulomki
# =====================================================================@001731=
# 1. podnaloga
# Izven razreda sestavite funkcijo `gcd(m, n)`, ki izračuna največji skupni
# delitelj števil `m` in `n`. Zgled:
# 
#     >>> gcd(35, 63)
#     7
# =============================================================================
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

# def gcd(a, b):
#     if b == 0:
#         return a
#     return gcd(b, a % b)

# =====================================================================@001732=
# 2. podnaloga
# Definirajte razred `Ulomek`, s katerim predstavimo ulomek. Števec in
# imenovalec sta celi števili, pri čemer je morebiten negativen predznak
# vedno v števcu. Ulomki naj bodo vedno okrajšani. Atributa naj se
# imenujeta `st` in `im`.
# 
# Najprej definirajte konstruktor `__init__(self, st, im)`. Zgled:
# 
#     >>> u = Ulomek(5, 20)
#     >>> u.st
#     1
#     >>> u.im
#     4
# =============================================================================

class Ulomek:
    def __init__(self, st, im):
        delitelj = gcd(st, im)
        self.st = st // delitelj
        self.im = im // delitelj

# =====================================================================@001733=
# 3. podnaloga
# Definirajte metodo  `__str__`, ki predstavi ulomek z nizom
# oblike `'st/im'`. Zgled:
# 
#     >>> u = Ulomek(5, 20)
#     >>> print(u)
#     1/4
# =============================================================================

    def __str__(self):
        return "{}/{}".format(self.st, self.im)

# =====================================================================@001734=
# 4. podnaloga
# Definirajte še metodo  `__repr__`, ki predstavi ulomek z nizom
# oblike `'Ulomek(st, im)'`. Zgled:
# 
#     >>> u = Ulomek(5, 20)
#     >>> u
#     Ulomek(1, 4)
# =============================================================================

    def __repr__(self):
        return "Ulomek({}, {})".format(self.st, self.im)

# =====================================================================@001735=
# 5. podnaloga
# Definirajte metodo  `__eq__(self, other)`, ki vrne `True` če sta dva
# ulomka enaka, in `False` sicer. Zgled:
# 
#     >>> Ulomek(1, 3) == Ulomek(2, 3)
#     False
#     >>> Ulomek(2, 3) == Ulomek(10, 15)
#     True
# =============================================================================
    def __eq__(self, other):
        return self.im == other.im and self.st == other.st

# =====================================================================@001736=
# 6. podnaloga
# Definirajte metodo  `__add__(self, other)`, ki vrne vsoto dveh ulomkov.
# Ko definirate to metodo, lahko ulomke seštevate kar z operatorjem `+`.
# Na primer:
# 
#     >>> Ulomek(1, 6) + Ulomek(1, 4)
#     Ulomek(5, 12)
# =============================================================================

    def __add__(self, other):
        return Ulomek(self.st * other.im + other.st * self.im, self.im * other.im)

# =====================================================================@001737=
# 7. podnaloga
# Definirajte metodo  `__sub__`, ki vrne razliko dveh ulomkov.
# Ko definirate to metodo, lahko ulomke odštevate kar z operatorjem `-`.
# Na primer:
# 
#     >>> Ulomek(1, 4) - Ulomek(1, 6)
#     Ulomek(1, 12)
# =============================================================================
    def __sub__(self, other):
        return Ulomek(self.st * other.im - other.st * self.im, self.im * other.im)


# =====================================================================@001738=
# 8. podnaloga
# Definirajte metodo  `__mul__`, ki vrne zmnožek dveh ulomkov.
# Ko definirate to metodo, lahko ulomke množite kar z operatorjem `*`.
# Na primer:
# 
#     >>> Ulomek(1, 3) * Ulomek(1, 2)
#     Ulomek(1, 6)
# =============================================================================
    def __mul__(self, other):
        return Ulomek(self.st * other.st, self.im * other.im)

# =====================================================================@001739=
# 9. podnaloga
# Definirajte metodo  `__truediv__`, ki vrne kvocient dveh
# ulomkov. Ko definirate to metodo, lahko ulomke delite kar z operatorjem
# `/`. Na primer:
# 
#     >>> Ulomek(1, 6) / Ulomek(1, 4)
#     Ulomek(2, 3)
# =============================================================================
    def __truediv__(self, other):
        return Ulomek(self.st * other.im, self.im * other.st)

# =====================================================================@001740=
# 10. podnaloga
# Izven razreda `Ulomek` definirajte funkcijo `priblizek(n)`, ki vrne
# vsoto $$\frac{1}{0!} + \frac{1}{1!} + \frac{1}{2!} + … + \frac{1}{n!}.$$
# Funkcija naj uporablja razred `Ulomek`. Zgled:
# 
#     >>> priblizek(5)
#     Ulomek(163, 60)
# 
# Ali je izračunana vrednost blizu števila $e$?
# =============================================================================
def priblizek(n):
    vsota = Ulomek(1, 1)
    clen = 1
    prod = 1
    while clen <= n:
        vsota += Ulomek(1, prod)
        clen += 1
        prod *= clen
    return vsota




































































































# ============================================================================@

'Če vam Python sporoča, da je v tej vrstici sintaktična napaka,'
'se napaka v resnici skriva v zadnjih vrsticah vaše kode.'

'Kode od tu naprej NE SPREMINJAJTE!'


















































import json, os, re, sys, shutil, traceback, urllib.error, urllib.request


import io, sys
from contextlib import contextmanager

class VisibleStringIO(io.StringIO):
    def read(self, size=None):
        x = io.StringIO.read(self, size)
        print(x, end='')
        return x

    def readline(self, size=None):
        line = io.StringIO.readline(self, size)
        print(line, end='')
        return line


class Check:
    parts = None
    current_part = None
    part_counter = None

    @staticmethod
    def has_solution(part):
        return part['solution'].strip() != ''

    @staticmethod
    def initialize(parts):
        Check.parts = parts
        for part in Check.parts:
            part['valid'] = True
            part['feedback'] = []
            part['secret'] = []

    @staticmethod
    def part():
        if Check.part_counter is None:
            Check.part_counter = 0
        else:
            Check.part_counter += 1
        Check.current_part = Check.parts[Check.part_counter]
        return Check.has_solution(Check.current_part)

    @staticmethod
    def feedback(message, *args, **kwargs):
        Check.current_part['feedback'].append(message.format(*args, **kwargs))

    @staticmethod
    def error(message, *args, **kwargs):
        Check.current_part['valid'] = False
        Check.feedback(message, *args, **kwargs)

    @staticmethod
    def clean(x, digits=6, typed=False):
        t = type(x)
        if t is float:
            x = round(x, digits)
            # Since -0.0 differs from 0.0 even after rounding,
            # we change it to 0.0 abusing the fact it behaves as False.
            v = x if x else 0.0
        elif t is complex:
            v = complex(Check.clean(x.real, digits, typed), Check.clean(x.imag, digits, typed))
        elif t is list:
            v = list([Check.clean(y, digits, typed) for y in x])
        elif t is tuple:
            v = tuple([Check.clean(y, digits, typed) for y in x])
        elif t is dict:
            v = sorted([(Check.clean(k, digits, typed), Check.clean(v, digits, typed)) for (k, v) in x.items()])
        elif t is set:
            v = sorted([Check.clean(y, digits, typed) for y in x])
        else:
            v = x
        return (t, v) if typed else v

    @staticmethod
    def secret(x, hint=None, clean=None):
        clean = Check.get('clean', clean)
        Check.current_part['secret'].append((str(clean(x)), hint))

    @staticmethod
    def equal(expression, expected_result, clean=None, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get('clean', clean)
        actual_result = eval(expression, global_env)
        if clean(actual_result) != clean(expected_result):
            Check.error('Izraz {0} vrne {1!r} namesto {2!r}.',
                        expression, actual_result, expected_result)
            return False
        else:
            return True

    @staticmethod
    def approx(expression, expected_result, tol=1e-6, env=None, update_env=None):
        try:
            import numpy as np
        except ImportError:
            Check.error('Namestiti morate numpy.')
            return False
        if not isinstance(expected_result, np.ndarray):
            Check.error('Ta funkcija je namenjena testiranju za tip np.ndarray.')

        if env is None:
            env = dict()
        env.update({'np': np})
        global_env = Check.init_environment(env=env, update_env=update_env)
        actual_result = eval(expression, global_env)
        if type(actual_result) is not type(expected_result):
            Check.error("Rezultat ima napačen tip. Pričakovan tip: {}, dobljen tip: {}.",
                        type(expected_result).__name__, type(actual_result).__name__)
            return False
        exp_shape = expected_result.shape
        act_shape = actual_result.shape
        if exp_shape != act_shape:
            Check.error("Obliki se ne ujemata. Pričakovana oblika: {}, dobljena oblika: {}.", exp_shape, act_shape)
            return False
        try:
            np.testing.assert_allclose(expected_result, actual_result, atol=tol, rtol=tol)
            return True
        except AssertionError as e:
            Check.error("Rezultat ni pravilen." + str(e))
            return False

    @staticmethod
    def run(statements, expected_state, clean=None, env=None, update_env=None):
        code = "\n".join(statements)
        statements = "  >>> " + "\n  >>> ".join(statements)
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get('clean', clean)
        exec(code, global_env)
        errors = []
        for (x, v) in expected_state.items():
            if x not in global_env:
                errors.append('morajo nastaviti spremenljivko {0}, vendar je ne'.format(x))
            elif clean(global_env[x]) != clean(v):
                errors.append('nastavijo {0} na {1!r} namesto na {2!r}'.format(x, global_env[x], v))
        if errors:
            Check.error('Ukazi\n{0}\n{1}.', statements,  ";\n".join(errors))
            return False
        else:
            return True

    @staticmethod
    @contextmanager
    def in_file(filename, content, encoding=None):
        encoding = Check.get('encoding', encoding)
        with open(filename, 'w', encoding=encoding) as f:
            for line in content:
                print(line, file=f)
        old_feedback = Check.current_part['feedback'][:]
        yield
        new_feedback = Check.current_part['feedback'][len(old_feedback):]
        Check.current_part['feedback'] = old_feedback
        if new_feedback:
            new_feedback = ['\n    '.join(error.split('\n')) for error in new_feedback]
            Check.error('Pri vhodni datoteki {0} z vsebino\n  {1}\nso se pojavile naslednje napake:\n- {2}', filename, '\n  '.join(content), '\n- '.join(new_feedback))

    @staticmethod
    @contextmanager
    def input(content, visible=None):
        old_stdin = sys.stdin
        old_feedback = Check.current_part['feedback'][:]
        try:
            with Check.set_stringio(visible):
                sys.stdin = Check.get('stringio')('\n'.join(content) + '\n')
                yield
        finally:
            sys.stdin = old_stdin
        new_feedback = Check.current_part['feedback'][len(old_feedback):]
        Check.current_part['feedback'] = old_feedback
        if new_feedback:
            new_feedback = ['\n  '.join(error.split('\n')) for error in new_feedback]
            Check.error('Pri vhodu\n  {0}\nso se pojavile naslednje napake:\n- {1}', '\n  '.join(content), '\n- '.join(new_feedback))

    @staticmethod
    def out_file(filename, content, encoding=None):
        encoding = Check.get('encoding', encoding)
        with open(filename, encoding=encoding) as f:
            out_lines = f.readlines()
        equal, diff, line_width = Check.difflines(out_lines, content)
        if equal:
            return True
        else:
            Check.error('Izhodna datoteka {0}\n  je enaka{1}  namesto:\n  {2}', filename, (line_width - 7) * ' ', '\n  '.join(diff))
            return False

    @staticmethod
    def output(expression, content, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(expression, global_env)
        finally:
            output = sys.stdout.getvalue().rstrip().splitlines()
            sys.stdout = old_stdout
        equal, diff, line_width = Check.difflines(output, content)
        if equal:
            return True
        else:
            Check.error('Program izpiše{0}  namesto:\n  {1}', (line_width - 13) * ' ', '\n  '.join(diff))
            return False

    @staticmethod
    def difflines(actual_lines, expected_lines):
        actual_len, expected_len = len(actual_lines), len(expected_lines)
        if actual_len < expected_len:
            actual_lines += (expected_len - actual_len) * ['\n']
        else:
            expected_lines += (actual_len - expected_len) * ['\n']
        equal = True
        line_width = max(len(actual_line.rstrip()) for actual_line in actual_lines + ['Program izpiše'])
        diff = []
        for out, given in zip(actual_lines, expected_lines):
            out, given = out.rstrip(), given.rstrip()
            if out != given:
                equal = False
            diff.append('{0} {1} {2}'.format(out.ljust(line_width), '|' if out == given else '*', given))
        return equal, diff, line_width

    @staticmethod
    def init_environment(env=None, update_env=None):
        global_env = globals()
        if not Check.get('update_env', update_env):
            global_env = dict(global_env)
        global_env.update(Check.get('env', env))
        return global_env

    @staticmethod
    def generator(expression, expected_values, should_stop=None, further_iter=None, clean=None, env=None, update_env=None):
        from types import GeneratorType
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get('clean', clean)
        gen = eval(expression, global_env)
        if not isinstance(gen, GeneratorType):
            Check.error("Izraz {0} ni generator.", expression)
            return False

        try:
            for iteration, expected_value in enumerate(expected_values):
                actual_value = next(gen)
                if clean(actual_value) != clean(expected_value):
                    Check.error("Vrednost #{0}, ki jo vrne generator {1} je {2!r} namesto {3!r}.",
                                iteration, expression, actual_value, expected_value)
                    return False
            for _ in range(Check.get('further_iter', further_iter)):
                next(gen)  # we will not validate it
        except StopIteration:
            Check.error("Generator {0} se prehitro izteče.", expression)
            return False

        if Check.get('should_stop', should_stop):
            try:
                next(gen)
                Check.error("Generator {0} se ne izteče (dovolj zgodaj).", expression)
            except StopIteration:
                pass  # this is fine
        return True

    @staticmethod
    def summarize():
        for i, part in enumerate(Check.parts):
            if not Check.has_solution(part):
                print('{0}. podnaloga je brez rešitve.'.format(i + 1))
            elif not part['valid']:
                print('{0}. podnaloga nima veljavne rešitve.'.format(i + 1))
            else:
                print('{0}. podnaloga ima veljavno rešitev.'.format(i + 1))
            for message in part['feedback']:
                print('  - {0}'.format('\n    '.join(message.splitlines())))

    settings_stack = [{
        'clean': clean.__func__,
        'encoding': None,
        'env': {},
        'further_iter': 0,
        'should_stop': False,
        'stringio': VisibleStringIO,
        'update_env': False,
    }]

    @staticmethod
    def get(key, value=None):
        if value is None:
            return Check.settings_stack[-1][key]
        return value

    @staticmethod
    @contextmanager
    def set(**kwargs):
        settings = dict(Check.settings_stack[-1])
        settings.update(kwargs)
        Check.settings_stack.append(settings)
        try:
            yield
        finally:
            Check.settings_stack.pop()

    @staticmethod
    @contextmanager
    def set_clean(clean=None, **kwargs):
        clean = clean or Check.clean
        with Check.set(clean=(lambda x: clean(x, **kwargs))
                             if kwargs else clean):
            yield

    @staticmethod
    @contextmanager
    def set_environment(**kwargs):
        env = dict(Check.get('env'))
        env.update(kwargs)
        with Check.set(env=env):
            yield

    @staticmethod
    @contextmanager
    def set_stringio(stringio):
        if stringio is True:
            stringio = VisibleStringIO
        elif stringio is False:
            stringio = io.StringIO
        if stringio is None or stringio is Check.get('stringio'):
            yield
        else:
            with Check.set(stringio=stringio):
                yield


def _validate_current_file():
    def extract_parts(filename):
        with open(filename, encoding='utf-8') as f:
            source = f.read()
        part_regex = re.compile(
            r'# =+@(?P<part>\d+)=\s*\n' # beginning of header
            r'(\s*#( [^\n]*)?\n)+?'     # description
            r'\s*# =+\s*?\n'            # end of header
            r'(?P<solution>.*?)'        # solution
            r'(?=\n\s*# =+@)',          # beginning of next part
            flags=re.DOTALL | re.MULTILINE
        )
        parts = [{
            'part': int(match.group('part')),
            'solution': match.group('solution')
        } for match in part_regex.finditer(source)]
        # The last solution extends all the way to the validation code,
        # so we strip any trailing whitespace from it.
        parts[-1]['solution'] = parts[-1]['solution'].rstrip()
        return parts

    def backup(filename):
        backup_filename = None
        suffix = 1
        while not backup_filename or os.path.exists(backup_filename):
            backup_filename = '{0}.{1}'.format(filename, suffix)
            suffix += 1
        shutil.copy(filename, backup_filename)
        return backup_filename

    def submit_parts(parts, url, token):
        submitted_parts = []
        for part in parts:
            if Check.has_solution(part):
                submitted_part = {
                    'part': part['part'],
                    'solution': part['solution'],
                    'valid': part['valid'],
                    'secret': [x for (x, _) in part['secret']],
                    'feedback': json.dumps(part['feedback']),
                }
                if 'token' in part:
                    submitted_part['token'] = part['token']
                submitted_parts.append(submitted_part)
        data = json.dumps(submitted_parts).encode('utf-8')
        headers = {
            'Authorization': token,
            'content-type': 'application/json'
        }
        request = urllib.request.Request(url, data=data, headers=headers)
        response = urllib.request.urlopen(request)
        return json.loads(response.read().decode('utf-8'))

    def update_attempts(old_parts, response):
        updates = {}
        for part in response['attempts']:
            part['feedback'] = json.loads(part['feedback'])
            updates[part['part']] = part
        for part in old_parts:
            valid_before = part['valid']
            part.update(updates.get(part['part'], {}))
            valid_after = part['valid']
            if valid_before and not valid_after:
                wrong_index = response['wrong_indices'].get(str(part['part']))
                if wrong_index is not None:
                    hint = part['secret'][wrong_index][1]
                    if hint:
                        part['feedback'].append('Namig: {}'.format(hint))


    filename = os.path.abspath(sys.argv[0])
    file_parts = extract_parts(filename)
    Check.initialize(file_parts)

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzMxfQ:1jQQgf:jYAH5c5c39s9IpEA_ROC8n-wOK0'
        try:
            test_data = [
                ('gcd(63, 35)', 7),
                ('gcd(40, 35)', 5),
                ('gcd(40, 19)', 1),
                ('gcd(15, 69)', 3),
                ('gcd(12345, 6789)', 3),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzMyfQ:1jQQgf:ROx8o2w0tqKniB82eUSKVhBr3dE'
        try:
            test_data = [
                ('Ulomek(5, 1).st', 5),
                ('Ulomek(5, 1).im', 1),
                ('Ulomek(5, 20).st', 1),
                ('Ulomek(5, 20).im', 4),
                ('Ulomek(20, 6).st', 10),
                ('Ulomek(20, 6).im', 3),
                ('Ulomek(5, 7).st', 5),
                ('Ulomek(5, 7).im', 7),
                ('Ulomek(7, 5).st', 7),
                ('Ulomek(7, 5).im', 5),
                ('Ulomek(-7, 5).st', -7),
                ('Ulomek(-7, 5).im', 5),
                ('Ulomek(-7, -5).st', 7),
                ('Ulomek(-7, -5).im', 5),
                ('Ulomek(0, 7).st', 0),
                ('Ulomek(0, 7).im', 1),
                ('Ulomek(40, -60).im', 3),
                ('Ulomek(40, -60).st', -2),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzMzfQ:1jQQgf:581gb57mYxbj8xdWzgaMHueyPzo'
        try:
            test_data = [
                ('str(Ulomek(20, 6))', '10/3'),
                ('str(Ulomek(0, 113))', '0/1'),
                ('str(Ulomek(40, -60))', '-2/3'),
                ('str(Ulomek(5, 20))', '1/4'),
                ('str(Ulomek(5, 7))', '5/7'),
                ('str(Ulomek(7, 5))', '7/5'),
                ('str(Ulomek(-7, 5))', '-7/5'),
                ('str(Ulomek(7, -5))', '-7/5'),
                ('str(Ulomek(-7, -5))', '7/5'),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzM0fQ:1jQQgf:E2jUu9InhcBumQP9dHGbxGHxqxQ'
        try:
            test_data = [
                ('repr(Ulomek(20, 6))', 'Ulomek(10, 3)'),
                ('repr(Ulomek(0, 226))', 'Ulomek(0, 1)'),
                ('repr(Ulomek(40, -60))', 'Ulomek(-2, 3)'),
                ('repr(Ulomek(5, 20))', 'Ulomek(1, 4)'),
                ('repr(Ulomek(5, 7))', 'Ulomek(5, 7)'),
                ('repr(Ulomek(7, 5))', 'Ulomek(7, 5)'),
                ('repr(Ulomek(-7, 5))', 'Ulomek(-7, 5)'),
                ('repr(Ulomek(7, -5))', 'Ulomek(-7, 5)'),
                ('repr(Ulomek(-7, -5))', 'Ulomek(7, 5)'),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzM1fQ:1jQQgf:2t7M2z3TFkacBS31bomx-KP17xQ'
        try:
            test_data = [
                ('Ulomek(1, 3) == Ulomek(2, 3)', False),
                ('Ulomek(2, 3) == Ulomek(10, 15)', True),
                ('Ulomek(0, 3) == Ulomek(0, 2215)', True),
                ('Ulomek(20, 6) == Ulomek(10, 3)', True),
                ('Ulomek(-10, 3) == Ulomek(10, 3)', False),
                ('Ulomek(10, -3) == Ulomek(10, 3)', False),
                ('Ulomek(1, 4) == Ulomek(4, 1)', False),
                ('Ulomek(20, 6) == Ulomek(10, 3)', True),
                ('Ulomek(40, -60) == Ulomek(-2, 3)', True),
                ('Ulomek(5, 20) == Ulomek(1, 4)', True),
                ('Ulomek(5, 7) == Ulomek(5, 7)', True),
                ('Ulomek(7, 5) == Ulomek(7, 5)', True),
                ('Ulomek(-7, 5) == Ulomek(-7, 5)', True),
                ('Ulomek(7, -5) == Ulomek(-7, 5)', True),
                ('Ulomek(-7, -5) == Ulomek(7, 5)', True),
                ('Ulomek(999999999, 1000000000) == Ulomek(999999998, 999999999)', False),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzM2fQ:1jQQgf:SvDxIlxQhnG5BwfHHeSOzfTdrEo'
        try:
            test_data = [
                ('Ulomek(1, 6) + Ulomek(1, 4)', Ulomek(5, 12)),
                ('Ulomek(1, -6) + Ulomek(-1, 4)', Ulomek(-5, 12)),
                ('Ulomek(1, 6) + Ulomek(-1, 4)', Ulomek(-1, 12)),
                ('Ulomek(1, -6) + Ulomek(1, 4)', Ulomek(1, 12)),
                ('Ulomek(1, 6) + Ulomek(1, 6)', Ulomek(1, 3)),
                ('Ulomek(1, 6) + Ulomek(-1, 6)', Ulomek(0, 1)),
                ('Ulomek(60, 1) + Ulomek(-1, 60)', Ulomek(3599, 60)),
                ('Ulomek(1, 2014) + Ulomek(1, 2015)', Ulomek(4029, 4058210)),
                ('Ulomek(1, 2014) + Ulomek(1, -2015)', Ulomek(1, 4058210)),
                ('Ulomek(757, 3000) + Ulomek(743, 3000)', Ulomek(1, 2)),
                ('Ulomek(1009, 2022) + Ulomek(1013, 2022)', Ulomek(1, 1)),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzM3fQ:1jQQgf:N9IRizKGMqIBC5_JqFEL2TIq_G8'
        try:
            test_data = [
                ('Ulomek(1, 6) - Ulomek(1, 4)', Ulomek(-1, 12)),
                ('Ulomek(1, 4) - Ulomek(1, 6)', Ulomek(1, 12)),
                ('Ulomek(3, 6) - Ulomek(1, 6)', Ulomek(1, 3)),
                ('Ulomek(1, -6) - Ulomek(-1, 4)', Ulomek(1, 12)),
                ('Ulomek(1, 6) - Ulomek(-1, 4)', Ulomek(5, 12)),
                ('Ulomek(1, -6) - Ulomek(1, 4)', Ulomek(-5, 12)),
                ('Ulomek(1, 6) - Ulomek(1, 6)', Ulomek(0, 1)),
                ('Ulomek(1, 6) - Ulomek(-1, 6)', Ulomek(1, 3)),
                ('Ulomek(60, 1) - Ulomek(-1, 60)', Ulomek(3601, 60)),
                ('Ulomek(1, 2014) - Ulomek(1, 2015)', Ulomek(1, 4058210)),
                ('Ulomek(1, 2014) - Ulomek(1, -2015)', Ulomek(4029, 4058210)),
                ('Ulomek(757, 3000) - Ulomek(743, 3000)', Ulomek(7, 1500)),
                ('Ulomek(2003, 1980) - Ulomek(1013, 1980)', Ulomek(1, 2)),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzM4fQ:1jQQgf:RUi4FSbJKcqX_qjTKKRarVNuDLE'
        try:
            test_data = [
                ('Ulomek(1, 3) * Ulomek(1, 2)', Ulomek(1, 6)),
                ('Ulomek(1, 6) * Ulomek(1, 4)', Ulomek(1, 24)),
                ('Ulomek(4, 9) * Ulomek(3, 2)', Ulomek(2, 3)),
                ('Ulomek(1, -6) * Ulomek(-1, 4)', Ulomek(1, 24)),
                ('Ulomek(1, 6) * Ulomek(-1, 4)', Ulomek(-1, 24)),
                ('Ulomek(1, -6) * Ulomek(1, 4)', Ulomek(-1, 24)),
                ('Ulomek(757, 3000) * Ulomek(743, 3000)', Ulomek(562451, 9000000)),
                ('Ulomek(60, 1) * Ulomek(-1, 60)', Ulomek(-1, 1)),
                ('Ulomek(25857, 160930) * Ulomek(277970, 33813)', Ulomek(247, 187)),
                ('Ulomek(25857, 1) * Ulomek(277970, 1)', Ulomek(7187470290, 1)),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzM5fQ:1jQQgf:omu3MBD77sWehIyXC_gAMs-5dBk'
        try:
            test_data = [
                ('Ulomek(1, 6) / Ulomek(1, 4)', Ulomek(2, 3)),
                ('Ulomek(4, 9) / Ulomek(2, 3)', Ulomek(2, 3)),
                ('Ulomek(1, -6) / Ulomek(-1, 4)', Ulomek(2, 3)),
                ('Ulomek(1, 6) / Ulomek(-1, 4)', Ulomek(-2, 3)),
                ('Ulomek(1, -6) / Ulomek(1, 4)', Ulomek(-2, 3)),
                ('Ulomek(757, 3000) / Ulomek(743, 3000)', Ulomek(757, 743)),
                ('Ulomek(757, 3000) / Ulomek(3000, 743)', Ulomek(562451, 9000000)),
                ('Ulomek(60, 1) / Ulomek(-60, 1)', Ulomek(-1, 1)),
                ('Ulomek(160930, 25857) / Ulomek(277970, 33813)', Ulomek(187, 247)),
                ('Ulomek(25857, 1) / Ulomek(277970, 1)', Ulomek(25857, 277970)),
                ('Ulomek(25857, 1) / Ulomek(1, 277970)', Ulomek(7187470290, 1)),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzQwfQ:1jQQgf:-Dby_bz9IE3dA3jufr77cyVrNUU'
        try:
            test_data = [
                ('priblizek(3)', Ulomek(8, 3)),
                ('priblizek(1)', Ulomek(2, 1)),
                ('priblizek(0)', Ulomek(1, 1)),
                ('priblizek(5)', Ulomek(163, 60)),
                ('priblizek(10)', Ulomek(9864101, 3628800)),
                ('priblizek(20)', Ulomek(6613313319248080001, 2432902008176640000)),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    print('Shranjujem rešitve na strežnik... ', end="")
    try:
        url = 'https://www.projekt-tomo.si/api/attempts/submit/'
        token = 'Token 399ac0c2612f691d2deb5ce42b496b66c5afd2c1'
        response = submit_parts(Check.parts, url, token)
    except urllib.error.URLError:
        print('PRI SHRANJEVANJU JE PRIŠLO DO NAPAKE! Poskusite znova.')
    else:
        print('Rešitve so shranjene.')
        update_attempts(Check.parts, response)
        if 'update' in response:
            print('Updating file... ', end="")
            backup_filename = backup(filename)
            with open(__file__, 'w', encoding='utf-8') as f:
                f.write(response['update'])
            print('Previous file has been renamed to {0}.'.format(backup_filename))
            print('If the file did not refresh in your editor, close and reopen it.')
    Check.summarize()

if __name__ == '__main__':
    _validate_current_file()
