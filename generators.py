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

def num_root(x):
    return '\sqrt['+str(random.randint(2,10))+']{'+x+'}'

def frac_of(a, b):
    return r'\frac{'+a+'}{'+b+'}'

def pow_of(a,b):
    return a+'^{'+b+'}'

def ration_num():
    return frac_of(nat_num(), nat_num())

def rand_num():
    num = [nat_num, real_num, ration_num]
    return rand_root(random.choice(num)())

def rand_root(x):
    if rand_bool(.9):
        return num_root(x)
    return x

def lin_fun(x = None):
    if x is None:
        x = var()
    k = exist_or_not_fst(rand_num(), .2)
    c = exist_or_not_term(rand_num(), .2)
    return k + x + c

def square_fun(x = None):
    if x is None:
        x = var()
    a, b, c = exist_or_not_fst(rand_num(), .2), exist_or_not(rand_num(), .2), exist_or_not_term(rand_num(), .2)
    return a + pow_of(x, '2') + exist_or_not_term(b+x, .2) + c

def cube_fun(x = None):
    if x is None:
        x = var()
    a, b, c, d = exist_or_not_fst(rand_num(), .2), exist_or_not(rand_num(), .2), exist_or_not(rand_num(), .2), \
                 exist_or_not_term(rand_num(), .2)
    return a + pow_of(x, '3') + exist_or_not_term(b + pow_of(x, '2'), .2) + exist_or_not_term(c+x, .2) + d

def frac_lin_fun():
    x = var()
    return frac_of(lin_fun(x), lin_fun(x))

def poly_fun(x = None, get_n = False):
    if x is None:
        x = var()
    n = random.randint(4, 20)
    c = exist_or_not_fst(rand_num(), .2)
    res = c + pow_of(x, str(n))
    for i in range(n-1, 1, -1):
        c = exist_or_not(rand_num())
        res += exist_or_not_term(c + pow_of(x, str(i)), .7)
    c = exist_or_not(rand_num(), .2)
    res += exist_or_not_term(c + x, .7)
    res += exist_or_not_term(rand_num(), .7)
    return res if not get_n else '\sqrt['+str(n+1)+']{'+res+'}'

def ration_fun():
    x = var()
    return frac_of(poly_fun(x), poly_fun(x))

def irration_fun():
    x = var()
    if rand_bool():
        return frac_of(poly_fun(x, True), poly_fun(x, True))
    else:
        return poly_fun(x, True)

def log_fn(x):
    fn = [lin_fun, square_fun, cube_fun]
    return '\log_{'+str(random.randint(2,10))+'}('+str(random.choice(fn)(x))+')'

def pow_fn(x):
    return x+'^{\log_{'+str(random.randint(2,10))+'}('+str(real_num())+')}'

def trig_fn(x):
    fn = [lin_fun, square_fun, cube_fun]
    t = ['tan', 'sin', 'cos', 'arcsin', 'cosh', 'sec', 'arctan', 'sinh', 'cosh', 'arccos']
    return '\\'+str(random.choice(t))+'('+str(random.choice(fn)(x))+')'

def trans_fun():
    x = var()
    fn = [log_fn, pow_fn, trig_fn]
    n = random.randint(1,10)
    res = ''
    for i in range(n):
        res += str(random.choice(fn)(x))+pos_or_neg()
    return res[:len(res)-1]


# полиномы не более 3-х членов
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

# генерация рациональных и иррациональных функций
def ration_sqrt():
    a, b = random.randint(2,100), random.randint(2,100)
    if a == b and a % 2 == 0:
        a += 1
        b += 1
    else:
        if b > a:
            a, b = b, a
        if a % b != 0:
            c = a // b
            if c == 1:
                c = random.randint(2,9)
            b = a * c
    return '\sqrt['+str(b)+']{x^{'+str(a)+'}}'

def irration_sqrt():
    if rand_bool(0.2): # b > a
        a, b = random.randint(2, 100), random.randint(2, 100)
        if a == b and a % 2 != 0:
            a += 1
        if b > a:
            a, b = b, a
        return '\sqrt[' + str(a) + ']{x^{' + str(b) + '}}'
    else: # case sqrt(x^2) = |x|
        a = random.randint(1,50)*2
        return '\sqrt['+str(a)+']{x^{'+str(a)+'}}'

def ration_fun_with_sqrt():
    return ration_sqrt()
    res = ''
    for _ in range(1, random.randint(1,20)):
        res += ration_sqrt() + pos_or_neg()
    return res

def irration_fun_with_sqrt():
    return irration_sqrt()
    flag = False
    while not flag:
        res = ''
        for _ in range(1, random.randint(1, 20)):
            if rand_bool(0.7):
                flag = True
                res += irration_sqrt()
            else:
                res += ration_sqrt()
            res+= pos_or_neg()
        res += ration_sqrt()
    return res

def tex_exp(exp):
    return '$'+exp+'$\n'

def writer(title, fn, n):
    with open(title, 'w+') as f:
        for _ in range(n):
            f.writelines(tex_exp(fn()))

def main():
    writer("data/lin_fun.txt", lin_fun, 10000)
    writer("data/square_fun.txt", square_fun, 10000)
    writer("data/cube_fun.txt", cube_fun, 10000)
    writer("data/poly_fun.txt", poly_fun, 10000)
    writer("data/frac_lin_fun.txt", frac_lin_fun, 10000)
    writer("data/ration_fun.txt", ration_fun, 10000)
    writer("data/irration_fun.txt", irration_fun, 10000)
    writer("data/trans_fun.txt", trans_fun, 10000)
    # writer("data/test/ration_fun.txt", ration_fun_with_sqrt, 20000)
    # writer("data/test/irration_fun.txt", irration_fun_with_sqrt, 20000)


if __name__ == '__main__':
    main()
