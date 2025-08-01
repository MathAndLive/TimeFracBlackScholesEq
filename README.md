# Проект БММ "Фонд «Институт “Вега”»: дробное по времени уравнение Блэка—Шоулза"

* **Geometric Brownian Motion (GBM)** — модель, описывающая поведение цены актива:

  $\displaystyle{dS_t = \mu S_t dt + \sigma S_t dW_t}$

  где:

  * $S_t$ — цена актива в момент времени $t$,
  * $\mu$ — средняя доходность, 
  * $\sigma$ — волатильность (стандартное отклонение лог-доходности),
  * $W_t$ — стандартный винеровский процесс.

  Такое уравнение имеет решение:

  $\displaystyle{S_t = S_0 \exp\left((\mu - \tfrac{1}{2} \sigma^2)t + \sigma W_t\right)}$

* **Black-Scholes PDE**: уравнение на цену опциона $V(S, t)$

  $\displaystyle{\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0, \quad \tau = \inf \{s : Z_s = T \},}$

где $X_t$ — геометрическое броуновское движение, $Z_t$ — α-устойчивый субординатор.
