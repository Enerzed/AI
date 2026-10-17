#  Z = (Y-3)*sin(X) + (Y+3)*cos(X+Y),  X,Y [-12, 2]
#
#    - отбор: стохастическая универсальная выборка (SUS)
#    - скрещивание: одноточечное
#    - мутация: инверсия битов
#    - элитизм: 1 лучшая особь переходит в след. поколение

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# НАСТРОЙКИ
BITS      = 16
POP_SIZE  = 10
P_MUT     = 0.02
N_ELITE   = 1

X_MIN, X_MAX = -12.0, 2.0
Y_MIN, Y_MAX = -12.0, 2.0

# КОД ГРЕЯ
def gray_to_bin(g):
    #Код Грея -> обычное целое.
    b = g
    shift = 1
    while shift < BITS:
        b ^= b >> shift
        shift <<= 1
    return b


# Хромосома (массив бит) -> вещественное число в [lo, hi].
def decode(chrom, lo, hi):
    # биты -> целое
    g = 0
    for b in chrom:
        g = (g << 1) | int(b)
    # код Грея -> обычное целое
    v = gray_to_bin(g)
    # целое -> вещественное
    return lo + (hi - lo) * v / (2 ** BITS - 1)


# ЦЕЛЕВАЯ ФУНКЦИЯ
def f(x, y):
    return (y - 3) * np.sin(x) + (y + 3) * np.cos(x + y)


def fitness(pop_x, pop_y):
    return np.array([
        f(decode(pop_x[i], X_MIN, X_MAX),
          decode(pop_y[i], Y_MIN, Y_MAX))
        for i in range(POP_SIZE)
    ])


def coords_of(pop_x, pop_y):
    return np.array([
        [decode(pop_x[i], X_MIN, X_MAX),
         decode(pop_y[i], Y_MIN, Y_MAX)]
        for i in range(POP_SIZE)
    ])


# ОПЕРАТОРЫ 
def sus_selection(fit, n_select):
    # Стохастическая универсальная выборка.
    fmin = fit.min()
    shifted = fit - fmin + 1e-9 if fmin < 0 else fit + 1e-9
    probs = shifted / shifted.sum()
    cum = np.cumsum(probs)

    pointers = (np.arange(n_select) + np.random.random()) / n_select
    idx = np.searchsorted(cum, pointers)
    return np.clip(idx, 0, len(fit) - 1)


def one_point_crossover(p1, p2):
    point = np.random.randint(1, BITS)
    child1 = np.concatenate([p1[:point], p2[point:]])
    child2 = np.concatenate([p2[:point], p1[point:]])
    return child1, child2


def bit_flip_mutation(chrom):
    chances = np.random.random(16) # Генерирует 16 чисел от 0 до 1
    mask = chances < P_MUT # массив True False
    c = chrom.copy()
    c[mask] = 1 - c[mask]
    return c


# ШАГ ГА
def make_step(pop_x, pop_y):
    fit = fitness(pop_x, pop_y)

    # элитизм
    best_idx = np.argsort(fit)[-N_ELITE:]
    elite_x = pop_x[best_idx].copy()
    elite_y = pop_y[best_idx].copy()

    # отбор родителей
    parents = sus_selection(fit, POP_SIZE)

    new_x = np.empty_like(pop_x)
    new_y = np.empty_like(pop_y)

    new_x[:N_ELITE] = elite_x
    new_y[:N_ELITE] = elite_y

    i = N_ELITE
    while i < POP_SIZE:
        a = parents[i]
        b = parents[(i + 1) % POP_SIZE]

        cx1, cx2 = one_point_crossover(pop_x[a], pop_x[b])
        cy1, cy2 = one_point_crossover(pop_y[a], pop_y[b])

        new_x[i] = bit_flip_mutation(cx1)
        new_y[i] = bit_flip_mutation(cy1)
        i += 1

        if i < POP_SIZE:
            new_x[i] = bit_flip_mutation(cx2)
            new_y[i] = bit_flip_mutation(cy2)
            i += 1

    return new_x, new_y


# НАЧАЛЬНАЯ ПОПУЛЯЦИЯ
def random_population():
    px = np.random.randint(0, 2, size=(POP_SIZE, BITS)).astype(np.uint8)
    py = np.random.randint(0, 2, size=(POP_SIZE, BITS)).astype(np.uint8)
    return px, py


pop_x, pop_y = random_population()
iteration = 0

# ПОВЕРХНОСТЬ
xs = np.linspace(X_MIN, X_MAX, 100)
ys = np.linspace(Y_MIN, Y_MAX, 100)
Xg, Yg = np.meshgrid(xs, ys)
Zg = f(Xg, Yg)

# ГРАФИК
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.18)

ax.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.7,
                rstride=2, cstride=2, edgecolor='none')

fit = fitness(pop_x, pop_y)
pts = coords_of(pop_x, pop_y)
best = int(np.argmax(fit))

scatter = ax.scatter(pts[:, 0], pts[:, 1], fit,
                     c='red', s=45, depthshade=False, label='Популяция')
best_pt = ax.scatter([pts[best, 0]], [pts[best, 1]], [fit[best]],
                     c='yellow', s=130, marker='D', depthshade=False,
                     edgecolors='k', label='Лучший')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Итерация 0 | best Z = {fit[best]:.4f}')
ax.legend(loc='upper left')


# ПЕРЕРИСОВКА
def redraw():
    global fit, pts, best
    fit = fitness(pop_x, pop_y)
    pts = coords_of(pop_x, pop_y)
    best = int(np.argmax(fit))

    scatter._offsets3d = (pts[:, 0], pts[:, 1], fit)
    best_pt._offsets3d = ([pts[best, 0]], [pts[best, 1]], [fit[best]])

    ax.set_title(
        f'Итерация {iteration} | best Z = {fit[best]:.4f} '
        f'(X={pts[best, 0]:.3f}, Y={pts[best, 1]:.3f})'
    )
    fig.canvas.draw_idle()


def on_next(event):
    global pop_x, pop_y, iteration
    pop_x, pop_y = make_step(pop_x, pop_y)
    iteration += 1
    redraw()


def on_reset(event):
    global pop_x, pop_y, iteration
    pop_x, pop_y = random_population()
    iteration = 0
    redraw()


# КНОПКИ
ax_next  = plt.axes([0.68, 0.05, 0.14, 0.05])
ax_reset = plt.axes([0.83, 0.05, 0.14, 0.05])

btn_next  = Button(ax_next,  'Next iteration')
btn_next.on_clicked(on_next)

btn_reset = Button(ax_reset, 'Reset')
btn_reset.on_clicked(on_reset)

plt.show()

# ИТОГ В КОНСОЛЬ
print(f"Итераций: {iteration}")
print(f"Лучшее: X={pts[best, 0]:.6f}, Y={pts[best, 1]:.6f}, Z={fit[best]:.6f}")