# =============================================================================
# Polinomi
#
# Definirajte razred `Polinom`, s katerim predstavimo polinom v
# spremenljivki $x$. Polinom predstavimo s seznamom njegovih koeficientov,
# kjer je $k$-ti element seznama koeficient pri $x^k$.
# 
# Na primer, polinom $x^3 + 2x + 7$ predstavimo s `Polinom([7, 2, 0, 1])`.
# Razmislite, kaj predstavlja `Polinom([])`. Zadnji koeficient v seznamu
# mora biti neničelen.
# =====================================================================@001741=
# 1. podnaloga
# Sestavite konstruktor `__init__(self, koef)`, ki nastavi objektu
# nastavi atribut `koef` (koeficienti polinoma). Zgled:
# 
#     >>> p = Polinom([7, 2, 0, 1])
#     >>> p.koef
#     [7, 2, 0, 1]
# 
# _Pozor_: Če kasneje spremenimo seznam, ki smo ga kot argument podali
# konstruktorju, se koeficienti polinoma ne smejo premeniti. Seznama,
# ki smo ga podali kot argument, konstruktor prav tako ne sme spremeninjati.
# Zgled:
# 
#     >>> l = [2, 0, 1]
#     >>> p = Polinom(l)
#     >>> l.append(3)
#     >>> p.koef
#     [2, 0, 1]
# =============================================================================
import math

class Polinom:
    def __init__(self, koef):
        # kopiraj koeficiente
        self.koef = koef[::1]
        # Izlusci ničlne koeficiente.
        while len(self.koef) > 0 and self.koef[-1] == 0:
            self.koef.pop()
            


# =====================================================================@001742=
# 2. podnaloga
# Sestavite metodo `stopnja`, ki vrne stopnjo polinoma. Zgled:
# 
#     >>> p = Polinom([7, 2, 0, 1])
#     >>> p.stopnja()
#     3
# 
# _Opomba_: Za razpravo glede stopnje ničelnega polinoma glejte [članek
# na Wikipediji](http://en.wikipedia.org/wiki/Degree_of_a_polynomial#Degree_of_the_zero_polynomial).
# =============================================================================
    def stopnja(self):
        if len(self.koef) == 0:
            return float("-inf")
        return len(self.koef) - 1

# =====================================================================@001743=
# 3. podnaloga
# Sestavite metodo `__repr__`, ki vrne niz oblike
# `'Polinom([a_0, …, a_n])'`, kjer so `a_0, …, a_n` koeficienti polinoma.
# Zgled:
# 
#     >>> p = Polinom([5, 0, 1])
#     >>> p
#     Polinom([5, 0, 1])
# =============================================================================
    def __repr__(self):
        return "Polinom({})".format(str(self.koef))

# =====================================================================@001744=
# 4. podnaloga
# Sestavite metodo `__eq__(self, other)` za primerjanje polinomov. Zgled:
# 
#     >>> Polinom([3, 2, 0, 1]) == Polinom([3, 2])
#     False
#     >>> Polinom([3, 2, 1, 0]) == Polinom([3, 2, 1])
#     True
# =============================================================================
    def __eq__(self, other):
        return self.koef == other.koef

# =====================================================================@001745=
# 5. podnaloga
# Sestavite metodo `__call__(self, x)`, ki izračuna in vrne vrednost
# polinoma v `x`. Pri izračunu vrednosti uporabite Hornerjev algoritem.
# Če definiramo metodo `__call__`, objekt postane "klicljiv" (tj. lahko
# ga kličemo, kakor da bi bil funkcija). Zgled:
# 
#     >>> p = Polinom([3, 2, 0, 1])
#     >>> p(1)
#     6
#     >>> p(-3)
#     -30
#     >>> p(0.725)
#     4.8310781249999994
# =============================================================================
    def __call__(self, x):
        vrednost = 0
        for k in reversed(self.koef):
            vrednost = vrednost * x + k
        return vrednost

# =====================================================================@001746=
# 6. podnaloga
# Sestavite metodo `__add__(self, other)` za seštevanje polinomov. Metoda
# naj sestavi in vrne nov objekt razreda `Polinom`, ki bo vsota polinomov
# `self` in `other`. Zgled:
# 
#     >>> Polinom([1, 0, 1]) + Polinom([4, 2])
#     Polinom([5, 2, 1])
# 
# _Pozor_: Pri seštevanju se lahko zgodi, da se nekateri koeficienti
# pokrajšajo: $(x^3 + 2x + 7) + (-x^3 - 2x + 10) = 17$.
# =============================================================================
    def __add__(self, other):
        n = max(len(self.koef), len(other.koef))
        ks = []
        for i in range(n):
            if i >= len(self.koef):
                ks.append(other.koef[i])
            elif i >= len(other.koef):
                ks.append(self.koef[i])
            else:
                ks.append(self.koef[i] + other.koef[i])
        return Polinom(ks)

# =====================================================================@001747=
# 7. podnaloga
# Sestavite metodo `__mul__` za množenje polinomov. Metoda
# naj sestavi in vrne nov objekt razreda `Polinom`, ki bo produkt polinomov.
# Zgled:
# 
#     >>> Polinom([1, 0, 1]) * Polinom([4, 2])
#     Polinom([4, 2, 4, 2])
# =============================================================================
    def __mul__(self, other):
        vsota = Polinom([])
        # zmnozimo vsako potenco posebej
        for n in range(len(self.koef)):
            # zamaknemo za potenco
            zmnoz = [0] * n
            for i in range(len(other.koef)):
                zmnoz.append(self.koef[n] * other.koef[i])
            # pristejemo k vsoti
            vsota += Polinom(zmnoz)
        return vsota

# =====================================================================@001748=
# 8. podnaloga
# Sestavite metodo `odvod(self, k)`, sestavi in vrne nov polinom, ki bo
# $k$-ti odvod polinoma `self`. Argument `k` naj ima privzeto vrednost 1.
# Zgled:
# 
#     >>> p = Polinom([5, 1, 4, -3, 5, -1])
#     >>> p.odvod()
#     Polinom([1, 8, -9, 20, -5])
#     >>> p.odvod(2)
#     Polinom([8, -18, 60, -20])
# =============================================================================
    def odvod(self, k=1):
        koef_odovod = []
        for n in range(k, len(self.koef)):
            koef = math.factorial(n) // math.factorial(n - k)
            koef_odovod.append(koef * self.koef[n])
        return Polinom(koef_odovod)


# =====================================================================@001749=
# 9. podnaloga
# Sestavite metodo `__str__`, ki predstavi polinom v čitljivi obliki,
# kot kaže primer:
# 
#     >>> p = Polinom([5, 1, 4, -3, 5, -1])
#     -x^5 + 5x^4 - 3x^3 + 4x^2 + x + 5
# 
# Za niz, ki ga funkcija vrne, naj velja naslednje:
# 
# * Polinom je sestavljen iz monomov oblike `ax^k`, kjer je `a` ustrezen
#   koeficient.
# * Monomi so med seboj povezani z znaki `+`; pred in za plusom je po en
#   presledek.
# * Namesto `x^1` bomo pisali samo `x`, `x^0` pa bomo izpustili in pisali
#   samo koeficient.
# * Če je koeficient 1, bomo namesto `1x^k` pisali `x^k`. Če je -1, bomo
#   namesto `-1x^k` pisali `-x^k`. To ne velja za prosti člen.
# * Če je koeficient negativen, bomo pri združevanju monomov uporabili
#   znak `-` namesto znaka `+`. Torej, namesto `ax^m + -bx^n` bomo pisali
#   `ax^m - bx^n`. To ne velja za vodilni člen.
# * Če je koeficient 0, bomo monom izpustili. To pravilo ne velja za
#   ničelni polinom.
# =============================================================================
    def __str__(self):
        n = len(self.koef)
        # nicelni polinom
        if n == 0:
            return "0"
        niz = ""
        for i in range(n):
            # normalen poilnom
            stopnja = n - 1 - i
            vodilni = i == 0
            k = self.koef[- (i + 1)] # od najvecje stopnje proti najmanjsi
            if k == 0:
                continue
            # izvzemi predznak
            predznak = "" if vodilni and k >= 0 else "+" if k >= 0 else "-"
            k = str(abs(k)) if abs(k) != 1 or stopnja == 0 else ""
            # pravila
            if stopnja == 0:
                if vodilni:
                    niz += "{}{}".format(predznak, k)
                else:
                    niz += " {} {}".format(predznak, k)
            elif stopnja == 1:
                if vodilni:
                    niz += "{}{}x".format(predznak, k)
                else:
                    niz += " {} {}x".format(predznak, k)
            else:    
                if vodilni:
                    niz += "{}{}x^{}".format(predznak, k, stopnja)
                else:
                    niz += " {} {}x^{}".format(predznak, k, stopnja)
        return niz




































































































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
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzQxfQ:1jQRAh:vwaspryYo7bvpwcCDeM4bIUC8N8'
        try:
            test_data = [
                ('Polinom([1, 2, 3]).koef', [1, 2, 3]),
                ('Polinom([1, 2, 0, 0]).koef', [1, 2]),
                ('Polinom([0, 0, 0, 0]).koef', []),
                ('Polinom([0, 0, 0, 0, 0, 0, 0, 7]).koef', [0, 0, 0, 0, 0, 0, 0, 7]),
            ]
            additional_tests = [
                (["l = [1, 2, 3, 0, 0, 0]",
                  "p = Polinom(l)",
                  "k = p.koef"],
                 {'l': [1, 2, 3, 0, 0, 0],
                  'k': [1, 2, 3]}),
                (["l = [1, 2, 3, 0, 0, 0]",
                  "p = Polinom(l)",
                  "l.extend([13, 14, 15])",
                  "k = p.koef"],
                 {'l': [1, 2, 3, 0, 0, 0, 13, 14, 15],
                  'k': [1, 2, 3]}),
                (["l = [1, 2, 3]",
                  "p = Polinom(l)",
                  "del l[-1]",
                  "del l[-1]", 
                  "k = p.koef"],
                 {'l': [1],
                  'k': [1, 2, 3]}),
            ]
            vse_ok = True
            for td in test_data:
                if not Check.equal(*td):
                    vse_ok = False
                    break
            if vse_ok:
                for td in additional_tests:
                    if Check.run(*td):
                        break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzQyfQ:1jQRAh:au82LKbOrlUdrGHIeMwvqXqk-rU'
        try:
            test_data = [
                ('Polinom([7, 2, 0, 1]).stopnja()', 3),
                ('Polinom([1, 2, 3]).stopnja()', 2),
                ('Polinom([1, 2, 3, 4, 5, 13, -22]).stopnja()', 6),
                ('Polinom([1]).stopnja()', 0),
                ('Polinom([]).stopnja()', float('-inf')),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzQzfQ:1jQRAh:EcKY0gjOtygoGo4oEvgxlGcMmFY'
        try:
            test_data = [
                ('repr(Polinom([1, 2, 3]))', "Polinom([1, 2, 3])"),
                ('repr(Polinom([1, 2, 3, 0, 0]))', "Polinom([1, 2, 3])"),
                ('repr(Polinom([]))', "Polinom([])"),
                ('repr(Polinom([0, 0]))', "Polinom([])"),
                ('repr(Polinom([1, 3]))', "Polinom([1, 3])"),
                ('repr(Polinom([7]))', "Polinom([7])"),
                ('repr(Polinom([1, -2, 3, -1]))', "Polinom([1, -2, 3, -1])"),
                ('repr(Polinom([1, 0, 0, -1]))', "Polinom([1, 0, 0, -1])"),
                ('repr(Polinom([0, 0, 0, -5]))', "Polinom([0, 0, 0, -5])"),
                ('repr(Polinom([-1, 2, -3, 1]))', "Polinom([-1, 2, -3, 1])"),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzQ0fQ:1jQRAh:hsgk2X7VIvTCV1sblm16ytHsFh0'
        try:
            test_data = [
                ('Polinom([1, 2, 3]) == Polinom([1, 2, 3, 0, 0])', True),
                ('Polinom([3, 2, 1, 0]) == Polinom([3, 2])', False),
                ('Polinom([3, 2, 1, 0]) == Polinom([3, 2, 1])', True),
                ('Polinom([1, 2, 3]) != Polinom([0, 1, 2, 3])', True),
                ('Polinom([1, 2, 3]) == Polinom([3, 2, 1])', False),
                ('Polinom([]) == Polinom([3, 2, 1])', False),
                ('Polinom([]) == Polinom([0])', True),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzQ1fQ:1jQRAh:nBHZnC5RdQlHRoiPk8f3B9IfK08'
        try:
            test_data = [
                ('Polinom([3, 2, 0, 1])(1)', 6),
                ('Polinom([1, 1, 1, 1, 1, 1, 1, 1, 1])(0)', 1),
                ('Polinom([1, 1, 1, 1, 1, 1, 1, 1, 1])(1)', 9),
                ('Polinom([1, 1, 1, 1, 1, 1, 1, 1, 1])(3)', 9841),
                ('Polinom([3, 2, 0, 1])(-3)', -30),
                ('Polinom([3, 2, 0, 1])(0.725)', 4.8310781249999994),
                ('Polinom([])(1337)', 0),
                ('Polinom([])(-666)', 0),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzQ2fQ:1jQRAh:IIRuyaYbVw0n4h4xqUM0TvcHdIo'
        try:
            test_data = [
                ('Polinom([1, 2, 3, 0, 0, 0, 0, 7]) + Polinom([4, 5])', Polinom([5, 7, 3, 0, 0, 0, 0, 7])),
                ('Polinom([1, 2, 3]) + Polinom([4, 5, 0, 0, 0, 0, 0, 0, -1])', Polinom([5, 7, 3, 0, 0, 0, 0, 0, -1])),
                ('Polinom([1, 2, 3]) + Polinom([4, 5])', Polinom([5, 7, 3])),
                ('Polinom([1, 0, 1]) + Polinom([4, 2])', Polinom([5, 2, 1])),
                ('Polinom([1, 2, 3]) + Polinom([-1, -2])', Polinom([0, 0, 3])),
                ('Polinom([1, 2, 3]) + Polinom([0, 0, -3])', Polinom([1, 2])),
            ]
            additional_tests = [
                (["p = Polinom([1, 2, 3])",
                  "q = Polinom([1, 2, 3, 4, 5, 6, 7])",
                  "r = p + q",
                  "p_koef, q_koef = p.koef, q.koef"],
                 {'r': Polinom([2, 4, 6, 4, 5, 6, 7]),
                  'p_koef': [1, 2, 3],
                  'q_koef': [1, 2, 3, 4, 5, 6, 7]}),
            ]
            vse_ok = True
            for td in test_data:
                if not Check.equal(*td):
                    vse_ok = False
                    break
            if vse_ok:
                for td in additional_tests:
                    if Check.run(*td):
                        break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzQ3fQ:1jQRAh:tjNRZW3lpJHKvx1OAXlNmUI9ey8'
        try:
            test_data = [
                ('Polinom([1, 0, 1]) * Polinom([4, 2])', Polinom([4, 2, 4, 2])),
                ('Polinom([0, 0, 0, 0, 0, 0, 1]) * Polinom([0, 0, 0, 0, 0, 0, 1])',
                 Polinom([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])),
                ('Polinom([1, 0, 1]) * Polinom([])', Polinom([])),
                ('Polinom([]) * Polinom([4, 2])', Polinom([])),
                ('Polinom([]) * Polinom([])', Polinom([])),
                ('Polinom([1, 2, 3]) * Polinom([2])', Polinom([2, 4, 6])),
                ('Polinom([1, 2, 1]) * Polinom([1, 1])', Polinom([1, 3, 3, 1])),
                ('Polinom([1, 2, 1, 3, 1]) * Polinom([1, 1, 5, 4, 3, 1])', Polinom([1, 3, 8, 18, 20, 27, 22, 14, 6, 1])),
                ('Polinom([1, 2, 3]) * Polinom([0, 0, 0])', Polinom([])),
            ]
            additional_tests = [
                (["p = Polinom([1, 2, 3])",
                  "q = Polinom([1, 2, 3, 4, 5, 6, 7])",
                  "r = p * q",
                  "p_koef, q_koef = p.koef, q.koef"],
                 {'r': Polinom([1, 4, 10, 16, 22, 28, 34, 32, 21]),
                  'p_koef': [1, 2, 3],
                  'q_koef': [1, 2, 3, 4, 5, 6, 7]}),
            ]
            vse_ok = True
            for td in test_data:
                if not Check.equal(*td):
                    vse_ok = False
                    break
            if vse_ok:
                for td in additional_tests:
                    if Check.run(*td):
                        break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzQ4fQ:1jQRAh:o_AqWZlonYa1WvTImyEu5Q7n-vQ'
        try:
            test_data = [
                ('Polinom([5, 1, 4, -3, 5, -1]).odvod()', Polinom([1, 8, -9, 20, -5])),
                ('Polinom([5, 1, 4, -3, 5, -1]).odvod(2)', Polinom([8, -18, 60, -20])),
                ('Polinom([1, 2, 3]).odvod()', Polinom([2, 6])),
                ('Polinom([3, 2, 1, 0, 1, 2, 3]).odvod()', Polinom([2, 2, 0, 4, 10, 18])),
                ('Polinom([0, 0, 0, 0, 0, 0, 0, 0, 1]).odvod()',  Polinom([0, 0, 0, 0, 0, 0, 0, 8])),
                ('Polinom([5, 4, 3, 2, 1]).odvod()', Polinom([4, 6, 6, 4])),
                ('Polinom([1]).odvod()', Polinom([])),
                ('Polinom([]).odvod()', Polinom([])),
                ('Polinom([5, 4, 3, 2, 1]).odvod(0)', Polinom([5, 4, 3, 2, 1])),
                ('Polinom([1]).odvod(0)', Polinom([1])),
                ('Polinom([1, 2, 3]).odvod(2)', Polinom([6])),
                ('Polinom([3, 2, 1, 0, 1, 2, 3]).odvod(2)', Polinom([2, 0, 12, 40, 90])),
                ('Polinom([0, 0, 0, 0, 0, 0, 0, 0, 1]).odvod(2)', Polinom([0, 0, 0, 0, 0, 0, 56])),
                ('Polinom([1, 2, 3]).odvod(3)', Polinom([])),
                ('Polinom([3, 2, 1, 0, 1, 2, 3]).odvod(3)', Polinom([0, 24, 120, 360])),
                ('Polinom([0, 0, 0, 0, 0, 0, 0, 0, 1]).odvod(3)', Polinom([0, 0, 0, 0, 0, 336])),
                ('Polinom([0, 0, 0, 0, 0, 0, 0, 0, 1]).odvod(8)', Polinom([40320])),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoxNzQ5fQ:1jQRAh:8pi53aUWGuflmbSxNEVJVk-bPGI'
        try:
            test_data = [
                ('str(Polinom([1, 2, 3]))', "3x^2 + 2x + 1"),
                ('str(Polinom([-1, -2, -3]))', "-3x^2 - 2x - 1"),
                ('str(Polinom([-1, -1, 1]))', "x^2 - x - 1"),
                ('str(Polinom([1, 1, -1]))', "-x^2 + x + 1"),
                ('str(Polinom([1, 1, 1, 1, 1, 1, 1, 1]))', 'x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1'),
                ('str(Polinom([0, 0, 0, 0, 0, 0, 0, 1]))', 'x^7'),
                ('str(Polinom([5, 1, 4, -3, 5, -1]))', "-x^5 + 5x^4 - 3x^3 + 4x^2 + x + 5"),
                ('str(Polinom([1, 3]))', "3x + 1"),
                ('str(Polinom([1, -2, 3, -1]))', "-x^3 + 3x^2 - 2x + 1"),
                ('str(Polinom([1, 0, 0, -1]))', "-x^3 + 1"),
                ('str(Polinom([0, 0, 0, -5]))', "-5x^3"),
                ('str(Polinom([-1, 2, -3, 1]))', "x^3 - 3x^2 + 2x - 1"),
                ('str(Polinom([]))', "0"),
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
