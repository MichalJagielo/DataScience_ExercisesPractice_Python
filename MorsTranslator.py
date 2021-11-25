#Zadanie: Mamy dwie listy i tekst. Używając słownika stworzyć translator tekst na alfabet morsa. 

#Pomysł na rozwiązanie: Słownik z litery:mors. Porownanie elementow z tekst z elementami slownika.

#Wypisanie wymaganych wartosci do nowego slownika lub listy. 

#Wynik: Jak ustawie klucz w slownik na litere wypisuje wartości do nowego slownika obcinając znaki
#z powtarzajacych sie kluczy (logiczne). Jak zmienie slownik, że kluczem są znaki morsa - 
#nie przypisuje wartosci albo zwraca blad. Dla spacji przyjmujemy (litery " " = mors "/")

#HELP :)

litery = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U',
          'V','W','X','Y','Z',' ']
mors = ['•-','-•••','-•-•','-••','•','••-•','--•','••••','••','•---','-•-','•-••','--','-•',
        '---','•--•','--•-','•-•','•••','-','••-','•••-','•--','-••-','-•--','--••','/']
tekst = 'Matematyka i informatyka jest super'
print('--------------------------------------------------')
tekst1 = list(tekst.upper())        #Zamiana tekstu do translacji na wielkie litery
print(tekst1)
print('--------------------------------------------------')


slownik = {}        #Przypisanie klucz:wartosc z list litery[key]:mors[value]

for key in litery:
    for value in mors:
        slownik[key] = value
        mors.remove(value)
        break
print(slownik)

print("--------------------------------------------------")

print("Długość tekst1 =",len(tekst1))
print("Długość słownik =",len(slownik))
#x = list(slownik.keys())
#y = list(slownik.values())
#print(x)
#print(y)
print("--------------------------------------------------")

translator = {}   #Pusty slownik do ktorego przypiszemy wartosci z morsa == litery z tekstu


for el in tekst1:
    
    if el in litery:
        translator[el] = slownik[el] 
        
print(translator)
