# =============================================================================
# Geometrija
#
# Pri tej vaji bomo implementirali razrede `Vektor`, `Tocka` in `Premica`,
# ki predstavljajo vektor, točko in premico v evklidski ravnini. Nekaj
# metod je že implementiranih, nekaj pa jih boste implementirali sami.
# =====================================================================@002149=
# 1. podnaloga
# Kodo, objavljeno na strani http://pastebin.com/aFPm162n vnesite kot rešitev
# te podnaloge.
# =============================================================================
import math
 
# Konstante
EPS = 1e-12
 
# Pomožne funkcije
def eq(a, b, eps=EPS):
    return abs(a - b) < eps
 
class Vektor:
    """
   Vektor v ravnini.
   """
 
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
    def __eq__(self, other):
        """
       Vrni True, če sta vektorja enaka.
       """
        return eq(self.x, other.x) and eq(self.y, other.y)
 
    def __add__(self, other):
        """
       Vrni vsoto vektorjev self in other.
       """
        return Vektor(self.x + other.x, self.y + other.y)
 
    def skalarni_produkt(self, other):
        """
       Vrni standardni skalarni produkt vektorjev self in other.
       """
        return self.x * other.x + self.y * other.y
 
    def __mul__(self, other):
        """
       Vrni skalarni produkt vektorjev self in other, če je other tudi vektor.
       Sicer vrni produkt vektorja self s skalarjem other (če je other skalar).
       """
        if type(other) == Vektor:
            return self.skalarni_produkt(other)
        else:
            return Vektor(self.x * other, self.y * other)
 
    def normiran(self):
        """
       Vrni normiran vektor.
       """
        return self / abs(self)
 
 
class Tocka:
    """
   Točka v ravnini.
   """
 
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
    def __repr__(self):
        return 'Tocka({0}, {1})'.format(self.x, self.y)
 
    def __str__(self):
        return '({0}, {1})'.format(self.x, self.y)
 
    def __eq__(self, other):
        """
       Vrni True, če sta vektorja enaka.
       """
        return eq(self.x, other.x) and eq(self.y, other.y)
 
    def __sub__(self, other):
        """
       Vrni vektor od točke other do točke self.
       """
        return Vektor(self.x - other.x, self.y - other.y)
 
    def translacija(self, v):
        """
       Vrni translacijo točke self za vektor v.
       """
        return Tocka(self.x + v.x, self.y + v.y)
 
    def __add__(self, v):
        """
       Vrni translacijo točke self za vektor v.
       """
        return self.translacija(v)
 
    def razdalja_do(self, other):
        """
       Vrni razdaljo točke self do objekta other. Objekt other je lahko Tocka ali Premica.
       """
        if type(other) == Tocka:
            return abs(self - other)
        elif type(other) == Premica:
            return abs(other.predznacena_razdalja(self))
        else:
            raise TypeError("nepodprta operacija za tipa: '{0}' in '{1}'".format(type(self), type(other)))
 
 
class Premica:
    """
   Premica v ravnini.
   
   Premico predstavimo s točko in normalo.
   """
 
    def __init__(self, tocka, normala):
        self.tocka = tocka
        self.normala = normala
 
    def __repr__(self):
        return 'Premica({0}, {1})'.format(self.tocka, self.normala)
   
    def enacba(self):
        """
       Vrne koeficiente a, b in c za enačbo premice :math:`a x + b y = c`.
       """
        a, b = self.normala.x, self.normala.y
        c = self.normala.skalarni_produkt(self.tocka)
        return a, b, c
 
    def __str__(self):
        a, b, c = self.enacba()
        return '{0} x + {1} y = {2}'.format(a, b, c)
 
    def smernik(self):
        """
       Vrni vektor, ki leži na premici self.
       """
        a, b = self.normala.x, self.normala.y
        return Vektor(-b, a)
 
    def pravokotnica(self, tocka=None):
        tocka = tocka or self.tocka
        # Prejšnja vrstica je ekvivalentna tejle: if tocka is None: tocka = self.tocka
        return Premica(tocka, self.smernik())
 
    def predznacena_razdalja(self, tocka):
        """
       Vrni predznačeno razdaljo od točke tocka do premice self.
       """
        return self.normala.normiran() * (tocka - self.tocka)

# =====================================================================@002150=
# 2. podnaloga
# V razredu `Vektor` sestavite metodo `__repr__(self)`. Zgled:
# 
#     >>> v = Vektor(3, 2)
#     >>> v
#     Vektor(3, 2)
# 
# Obstoječemu razredu lahko dodate novo metodo takole (razred
# `FooBar` že obstaja):
# 
#     class FooBar(FooBar):
# 
#         def nova_metoda(self, baz):
#             pass
# 
# _Opomba:_ Če v interaktivni konzoli pokličemo nek objekt, se izpiše niz,
# ki ga vrne klic metode `__repr__` na tem objektu. Priporočilo je, da je
# niz, ki ga vrne metoda `__repr__`, veljavna programska koda v Pythonu,
# ki ustvari identično kopijo objekta.
# =============================================================================
class Vektor(Vektor):
    def __repr__(self):
        return f"Vektor({self.x}, {self.y})"

# =====================================================================@002151=
# 3. podnaloga
# V razredu `Vektor` sestavite metodo `__str__(self)`. Zgled:
# 
#     >>> v = Vektor(3, 2)
#     >>> print(v)
#     (3, 2)
# 
# _Opomba:_ Funkcija `print` na svojem argumentu pokliče metodo `__str__`
# in izpiše niz, ki ga ta metoda vrne. Metoda `__str__` običajno vrne
# razumljiv opis objekta, ki naj bi ga razumeli tudi ne-programerji.
# =============================================================================
class Vektor(Vektor):
    def __str__(self):
        return f"({self.x}, {self.y})"

# =====================================================================@002152=
# 4. podnaloga
# V razredu `Vektor` sestavite metodo `__abs__(self)`, ki naj vrne dolžino
# (normo) vektorja. Zgled:
# 
#     >>> v = Vektor(1, 3)
#     >>> abs(v)
#     3.1622776601683795
# =============================================================================
class Vektor(Vektor):
    def __abs__(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

# =====================================================================@002153=
# 5. podnaloga
# V razredu `Vektor` sestavite metodo `__sub__(self, other)`, ki vrne
# razliko vektorjev. Zgled:
# 
#     >>> v = Vektor(-1, 3)
#     >>> u = Vektor(2, 1)
#     >>> u - v
#     Vektor(-3, 2)
# =============================================================================
class Vektor(Vektor):
    def __sub__(self, other):
        return Vektor(self.x - other.x, self.y - other.y)

# =====================================================================@002154=
# 6. podnaloga
# V razredu `Vektor` sestavite metodo `__truediv__(self, skalar)`, ki vrne
# produkt vektorja `self` s skalarjem `1 / skalar`. Zgled:
# 
#     >>> Vektor(-1, 3) / 2
#     Vektor(-0.5, 1.5)
# =============================================================================
class Vektor(Vektor):
    def __truediv__(self, skalar):
        return Vektor(self.x / skalar, self.y / skalar)

# =====================================================================@002155=
# 7. podnaloga
# V razredu `Vektor` sestavite metodo `sta_pravokotna(self, other)`, ki
# vrne `True`, če sta vektorja `self` in `other` pravokotna, in `False`
# sicer. Zgled:
# 
#     >>> v = Vektor(-1, 3)
#     >>> u = Vektor(2, 1)
#     >>> v.sta_pravokotna(u)
#     False
# =============================================================================
class Vektor(Vektor):
    def sta_pravokotna(self, other):
        return self.skalarni_produkt(other) == 0
# =====================================================================@002156=
# 8. podnaloga
# V razredu `Vektor` sestavite metodo `rotacija(self, alpha)`, ki vrne
# rotacijo vektorja `self` za kot `alpha` (v radianih). Zgled:
# 
#     >>> Vektor(1, 0).rotacija(math.pi/4)
#     Vektor(0.7071067811865476, 0.7071067811865475)
# =============================================================================
class Vektor(Vektor):
    def rotacija(self, a):
        # rotacija s pomočjo rotacijske matrike
        # [cos a, -sin a]
        # [sin a,  cos a]
        # in evaluacije linerane preslikave na vektorju
        x = math.cos(a) * self.x - math.sin(a) * self.y
        y = math.sin(a) * self.x + math.cos(a) * self.y
        return Vektor(x, y)

# =====================================================================@002157=
# 9. podnaloga
# V razredu `Premica` sestavite metodo `projekcija(self, tocka)`, ki vrne
# pravokotno projekcijo točke `tocka` na premico `self`. Zgled:
# 
#     >>> p = Premica(Tocka(1, 1), Vektor(0, 1))
#     >>> p.projekcija(Tocka(3, 0))
#     Tocka(3, 1)
# =============================================================================
class Premica(Premica):
    def projekcija(self, tocka):
        s = self.smernik()
        v = tocka - self.tocka
        w = s * (v * s) / (s * s)
        return self.tocka.translacija(w)

# =====================================================================@002158=
# 10. podnaloga
# V razredu `Premica` sestavite metodo `presek(self, other)`, ki vrne
# točko, ki je presek dveh premic. Zgled:
# 
#     >>> p = Premica(Tocka(3, 4), Vektor(2, -1))
#     >>> q = Premica(Tocka(0, 1), Vektor(1, 2))
#     >>> p.presek(q)
#     Tocka(1.2, 0.4)
# =============================================================================
class Premica(Premica):
    def presek(self, other):
        (a1, b1, c1) = self.enacba()
        (a2, b2, c2) = other.enacba()
        x = (b2 * c1 - b1 * c2) / (a1 * b2 - a2 * b1)
        y = (a2 * c1 - a1 * c2) / (a2 * b1 - a1 * b2)
        return Tocka(x, y)


# =====================================================================@002159=
# 11. podnaloga
# V razredu `Tocka` sestavite metodo `zrcali_cez_premico(self, premica)`,
# ki vrne zrcalno sliko točke `self` čez premico `premica`. Zgled:
# 
#     >>> p = Premica(Tocka(1, 1), Vektor(0, 1))
#     >>> Tocka(3, 4).zrcali_cez_premico(p)
#     Tocka(3, -2)
# =============================================================================
class Tocka(Tocka):
    def zrcali_cez_premico(self, premica):
        proj = premica.projekcija(self)
        v = proj - self
        return self.translacija(v * 2)





































































































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
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoyMTQ5fQ:1jT17d:06TpaCSPbUCS7OD5V9_vZQuCsrA'
        try:
            pass
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoyMTUwfQ:1jT17d:OAWxZxju3M7YYyD9Z9E3nzkRitY'
        try:
            Check.equal('repr(Vektor(3, 2))', 'Vektor(3, 2)')
            Check.equal('repr(Vektor(1, 0.5))', 'Vektor(1, 0.5)')
            Check.equal('repr(Vektor(-1, 0.5))', 'Vektor(-1, 0.5)')
            Check.equal('repr(Vektor(1, -0.5))', 'Vektor(1, -0.5)')
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoyMTUxfQ:1jT17d:i-9H0SS_goE8feRzosI00V0sWUg'
        try:
            Check.equal('str(Vektor(3, 2))', '(3, 2)')
            Check.equal('str(Vektor(1, 0.5))', '(1, 0.5)')
            Check.equal('str(Vektor(-1, 0.5))', '(-1, 0.5)')
            Check.equal('str(Vektor(1, -0.5))', '(1, -0.5)')
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoyMTUyfQ:1jT17d:pE55LloZOzYr8oYkr3XVfTs4GBo'
        try:
            Check.equal('abs(Vektor(1, 3))', 3.1622776601683795)
            Check.equal('abs(Vektor(1, 0.5))', 1.118033988749895)
            Check.equal('abs(Vektor(-10, 3))', 10.44030650891055)
            Check.equal('abs(Vektor(1, -30))', 30.01666203960727)
            Check.equal('abs(Vektor(-6, -5))', 7.810249675906654)
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoyMTUzfQ:1jT17d:GnsI26UvplPAtk7KO8wojnU7Q2A'
        try:
            Check.equal('''Vektor(-1, 3) - Vektor(2, 1)''', Vektor(-3, 2))
            Check.equal('''Vektor(5, 3) - Vektor(2, 1)''', Vektor(3, 2))
            Check.equal('''Vektor(1, -3) - Vektor(2, 1)''', Vektor(-1, -4))
            Check.equal('''Vektor(1, 3) - Vektor(-2, 1)''', Vektor(3, 2))
            Check.equal('''Vektor(-1, 3) - Vektor(2, -1)''', Vektor(-3, 4))
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoyMTU0fQ:1jT17d:szRkFpgq2iiDyBs6ai_Xu2AdL3Q'
        try:
            Check.equal('''Vektor(-1, 3) / 2''', Vektor(-0.5, 1.5))
            Check.equal('''Vektor(-1, 3) / 7''', Vektor(-0.14285714285714285, 0.42857142857142855))
            Check.equal('''Vektor(2, 3) / 0.3''', Vektor(6.666666666666667, 10.0))
            Check.equal('''Vektor(5, 7) / 11''', Vektor(0.4545454545454546, 0.6363636363636364))
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoyMTU1fQ:1jT17d:OccRadgfpBjQ37FSEQyLuOqFNUc'
        try:
            Check.equal('''Vektor(-1, 3).sta_pravokotna(Vektor(2, 1))''', False)
            Check.equal('''Vektor(1, 3).sta_pravokotna(Vektor(3, 1))''', False)
            Check.equal('''Vektor(-1, 3).sta_pravokotna(Vektor(-3, -1))''', True)
            Check.equal('''Vektor(-2, 6).sta_pravokotna(Vektor(-3, -1))''', True)
            Check.equal('''Vektor(-1, 3).sta_pravokotna(Vektor(3, 1))''', True)
            Check.equal('''Vektor(-1, 3).sta_pravokotna(Vektor(6, 2))''', True)
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoyMTU2fQ:1jT17d:NcuGVm0nCxwO8Xzrhi0cW2o1pao'
        try:
            Check.equal('''Vektor(1, 0).rotacija(math.pi/4)''', Vektor(0.7071067811865476, 0.7071067811865475))
            Check.equal('''Vektor(0, 1).rotacija(math.pi/4)''', Vektor(-0.7071067811865476, 0.7071067811865475))
            Check.equal('''Vektor(2, 0).rotacija(math.pi/4)''', Vektor(1.4142135623730951, 1.4142135623730951))
            Check.equal('''Vektor(0, 2).rotacija(math.pi/4)''', Vektor(-1.4142135623730951, 1.4142135623730951))
            Check.equal('''Vektor(1, 0).rotacija(math.pi/2)''', Vektor(0.0, 1.0))
            Check.equal('''Vektor(1, 0).rotacija(math.pi)''', Vektor(-1.0, 0.0))
            Check.equal('''Vektor(1, 0).rotacija(3*math.pi/2)''', Vektor(0.0, -1.0))
            Check.equal('''Vektor(1, 0).rotacija(2*math.pi)''', Vektor(1, 0))
            Check.equal('''Vektor(3, 2).rotacija(0.5)''', Vektor(1.6738966084627123, 3.1934417395933545))
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoyMTU3fQ:1jT17d:M-z-958frYjEbX6zC6ICmFzf3dg'
        try:
            Check.equal('''Premica(Tocka(1, 1), Vektor(0, 1)).projekcija(Tocka(3, 0))''', Tocka(3.0, 1.0))
            Check.equal('''Premica(Tocka(0, 1), Vektor(1, 2)).projekcija(Tocka(3, 4))''', Tocka(1.2, 0.4))
            Check.equal('''Premica(Tocka(-2, -2), Vektor(0.410958904109589, -1.09589041095890)).projekcija(Tocka(0, 4))''', Tocka(1.72602739726027, -0.602739726027397))
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoyMTU4fQ:1jT17d:X3xsVhZdIzkOmq-bDLlKGe861IM'
        try:
            Check.equal('''Premica(Tocka(3, 4), Vektor(2, -1)).presek(Premica(Tocka(0, 1), Vektor(1, 2)))''', Tocka(1.2, 0.4))
            Check.equal('''Premica(Tocka(3, 4), Vektor(2, -1)).presek(Premica(Tocka(0, 1), Vektor(-1, -2)))''', Tocka(1.2, 0.4))
            Check.equal('''Premica(Tocka(2, 1), Vektor(-2, 4)).presek(Premica(Tocka(-3, 4), Vektor(6, 2)))''', Tocka(-1.428571428571429, -0.714285714285714))
            Check.equal('''Premica(Tocka(2, 1), Vektor(2, -4)).presek(Premica(Tocka(-2, 1), Vektor(6, 2)))''', Tocka(-1.428571428571429, -0.714285714285714))
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJ1c2VyIjo1MzU2LCJwYXJ0IjoyMTU5fQ:1jT17d:78zR-e5kJQ_ipdXOzLAKCL4iEcI'
        try:
            Check.equal('''Tocka(3, 4).zrcali_cez_premico(Premica(Tocka(1, 1), Vektor(0, 1)))''', Tocka(3, -2))
            Check.equal('''Tocka(3, 4).zrcali_cez_premico(Premica(Tocka(0, 1), Vektor(1, 2)))''', Tocka(-0.6, -3.2))
            Check.equal('''Tocka(0, 4).zrcali_cez_premico(Premica(Tocka(-2, -2), Vektor(0.410958904109589, -1.09589041095890)))''', Tocka(3.45205479452055, -5.20547945205479))
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
