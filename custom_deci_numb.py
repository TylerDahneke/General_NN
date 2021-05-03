from decimal import *


class cust_prec_deci:

    def __init__(self, numb, precision=25):
        getcontext().prec = precision
        self.numb = Decimal(numb)

    def mult(self, other):
        ph = self.numb * other.numb
        return cust_prec_deci(ph)

    def __repr__(self):
        return str(self.numb)


if __name__ == '__main__':
    first = .000000000000000000000000000000000000000000000000000000000000000000000000000000015
    second_deci = .000000000000000000000000000000000000000000000000000000000000000000000000000000015
    first_deci = cust_prec_deci(first)
    print(first_deci)
    print(second_deci)
