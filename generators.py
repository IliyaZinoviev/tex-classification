import random
from numpy.random import normal
import math

def pos_or_neg():
    return '+' if rand_bool() else '-'

def exist_or_not(c, p = .5):
    return c if rand_bool(p) else ''

def exist_or_not_term(c, p = .5):
    return exist_or_not(pos_or_neg() + c, p)

def exist_or_not_fst(c, p = .5):
    return exist_or_not(rand_neg(.8))+exist_or_not(c, p)

def var():
    return ''.join(random.choices(['x', 'y', 'z', 't', 'w', 'v'], k=1))

def rand_bool(c = .5):
    return random.random() >= c

def rand_neg(p = .5):
    return ('-' if rand_bool(p) else '')

def nat_num():
    n = 0
    while n == 0 or n == 1:
        n = round(10**random.uniform(0, 3))
    return str(n)

def real_num():
    r = 0
    while 0 == r:
        r = round(10*random.uniform(0, 3), math.ceil(abs(normal(1,4))))
    return str(r).replace('.', ',')

def frac_of(a, b):
    return r'\frac{'+a+'}{'+b+'}'

def pow_of(a,b):
    return a+'^{'+b+'}'

def ration_num():
    return frac_of(nat_num(), nat_num())

def rand_num():
    num = [nat_num, real_num, ration_num]
    return random.choice(num)()

def lin_fun(x = None):
    if x is None:
        x = var()
    k = exist_or_not_fst(rand_num(), .2)
    c = exist_or_not_term(rand_num(), .2)
    return k + x + c

def square_fun():
    x = var()
    a, b, c = exist_or_not_fst(rand_num(), .2), exist_or_not(rand_num(), .2), exist_or_not_term(rand_num(), .2)
    return a + pow_of(x, '2') + exist_or_not_term(b+x, .2) + c

def cube_fun():
    x = var()
    a, b, c, d = exist_or_not_fst(rand_num(), .2), exist_or_not(rand_num(), .2), exist_or_not(rand_num(), .2), \
                 exist_or_not_term(rand_num(), .2)
    return a + pow_of(x, '3') + exist_or_not_term(b + pow_of(x, '2'), .2) + exist_or_not_term(c+x, .2) + d

def frac_lin_fun():
    x = var()
    return frac_of(lin_fun(x), lin_fun(x))

def poly_fun(x = None):
    if x is None:
        x = var()
    n = random.randint(4, 100)
    c = exist_or_not_fst(rand_num(), .2)
    res = c + pow_of(x, str(n))
    for i in range(n-1, 1, -1):
        c = exist_or_not(rand_num())
        res += exist_or_not_term(c + pow_of(x, str(i)), .7)
    c = exist_or_not(rand_num(), .2)
    res += exist_or_not_term(c + x, .7)
    res += exist_or_not_term(rand_num(), .7)
    return res

def ration_fun():
    x = var()
    return frac_of(poly_fun(x), poly_fun(x))

def poly_fun_spec(x = None):
    if x is None:
        x = var()
    n = random.randint(4, 100)
    c = exist_or_not_fst(rand_num(), .2)
    res = c + pow_of(x, str(n))
    # for i in range(n-1, 1, -1):
    #     c = exist_or_not(rand_num())
    #     res += exist_or_not_term(c + pow_of(x, str(i)), .7)
    c = exist_or_not(rand_num(), .2)
    res += exist_or_not_term(c + x, .7)
    res += exist_or_not_term(rand_num(), .7)
    return res

def ration_fun_spec():
    x = var()
    return frac_of(poly_fun_spec(x), poly_fun_spec(x))

def tex_exp(exp):
    return '$'+exp+'$\n'

def writer(title, fn, n):
    with open(title, 'w+') as f:
        for _ in range(n):
            f.writelines(tex_exp(fn()))

def main():
    writer("data/test/lin_fun.txt", lin_fun, 1000)
    writer("data/test/square_fun.txt", square_fun, 1000)
    writer("data/test/cube_fun.txt", cube_fun, 1000)
    writer("data/test/poly_fun.txt", poly_fun_spec, 1000)
    writer("data/test/frac_lin_fun.txt", frac_lin_fun, 1000)
    writer("data/test/ration_fun.txt", ration_fun_spec, 1000)
if __name__ == '__main__':
    main()
