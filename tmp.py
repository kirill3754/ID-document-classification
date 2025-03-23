def f(a: list = []):
    # if a is None:
    #     a = []
    a.append(1)
    return a


if __name__ == "__main__":
    a = f()
    print(a)
    b = f()
    print(b)
    c = f()
    print(c)
