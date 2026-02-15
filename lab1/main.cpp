#include <iostream>

using namespace std;

int main()
{
    setlocale(LC_ALL, ".UTF-8");

    unsigned int n;
    cout << "Введите n: ";
    cin >> n;

    unsigned int a = 1, b;

    for(unsigned int i = 2; i <= n/2; i++) {
        if (n % i == 0) {
            a = n / i;
            break;
        }
    }

    b = n - a;

    cout << a << " " << b << endl;

    cout << "Бергер Денис Максимович, 090304-РПИа-025" << endl;

    cin.get();
    while (cin.get() != '\n');

    return 0;
}