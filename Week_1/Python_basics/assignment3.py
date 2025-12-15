def palindrome(b):
    if b[::-1] == b:
        print(b,"is a palindrome:)")
    else:
        print(b,"is not a palindrome:(")
a = input("Enter sequence:")
palindrome(a)