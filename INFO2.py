
#Définition de notre fonction principale
def dichotomie():

    
    #Partie où l'utilisateur devra entrer les diverses valeurs qu'utilisera notre fonction principale  
    print("Soit une fonction f définie sur un intervalle [a;b]")
    a = float(input("Entrez la valeur de a : "))
    b = float(input("Entrez la valeur de b : "))
    e = float(input("Entrez la valeur seuil e : "))
    
    
    x = a ;
    y = b ;
    i = 0 ;

    
    #Définition de la fonction qui demandera à l'utilisateur d'entrer l'expression de sa fonction
    def fct():
        f = input("Veuillez entrer votre fonction, en fonction de x ")
        return lambda x: eval(f)

    f = fct() #Stackage de la fonction entrée par l'ulisateur dans la variable f 

    print(" \nx = ", x,"\n y = ", y, "\n i = ", i)
    while(f(x)*f(x+e) > 0) :
        x = x+e
        s= x+ e/2
        i = i+1
    print("x = ", x+e)
    print("\n s = ", x + e/2)
    print(" \nx = ", x,"\n y = ", y, "\n i = ", i)
    
    print("\n \n  On obtient :")
    print("x = ", x+e)
    print("\n et  s = ", x + e/2)
    print(" \nx = ", x,"\n y = ", y, "\n i = ", i)
    
    return print("\n Calcul Termine !!!... \nUne valeur approchée de la racine  de  notre fonction à", e ,"près est :", s); # On retourne la valeur de S

if __name__=="main" :
    print(dichotomie()) # Affichage du resultat final retouné par la fonction principale













#Définition de notre fonction principale
def Balayage():

    from scipy.optimize import brentq
    #Partie où l'utilisateur devra entrer les diverses valeurs qu'utilisera notre fonction principale  
    print("Soit une fonction f définie sur un intervalle [a;b]")
    a = float(input("Entrez la valeur de a : "))
    b = float(input("Entrez la valeur de b : "))
    e = float(input("Entrez la valeur seuil e : "))
    
    
    x = a ;
    y = b ;
    S = x ;



    #Définition de la fonction qui demandera à l'utilisateur d'entrer l'expression de sa fonction
    def donner_fonct():
        f = input("Veuillez entrer votre fonction, en fonction de x ")
        return lambda x: eval(f)


    
     
    f = donner_fonct() #Stackage de la fonction entrée par l'ulisateur dans la variable f 

    print(" \nx = ", x,"\n y = ", y, "\n S = ", S)

    


    
    print("\n Les valeurs des images sont : \nf(",x,") = ", f(x))  #Affichage des valeurs de f(x) 
    
    if f(x)== 0 :
        S = y
        print(" x = ", x,"\n y = ", y, "\n S = ", S)

    
    while ((abs(y-x) >= e) and ( f(x)*f(y)< 0)): #La boucle "Tantque" avec les différentes conditions
        
        print("f(",x,") *  f(",y,") = ", f(x)*f(y)," et  |y-x| = ",abs(y-x) )
        S = (x+y)/2
        
        if (f(x)*f(S) < 0):
            y = S
            
        else:
            x = S
        print(" x = ", x,"\n y = ", y, "\n S = ", S)


    print("\n\n On a en fin : ")
    print("f(",x,") *  f(",y,") = ", f(x)*f(y)," et  |y-x| = ",abs(y-x) )
    print(" x = ", x,"\n y = ", y, "\n S = ", S,"\n")


    print("Avec la fonction déjà créée par les developpeurs on a :",brentq(f,a,b))
        
    return print("\n OUFF!! Terminé... \n *=============================================================================* \nLa valeur approchée de votre fonction à", e ,"près est :", S); # On retourne la valeur de S

if __name__=="main" :
    print(Balayage()) # Affichage du resultat final retouné par la fonction principale


















#On défini notre fonction principale 'Newton'
def Newton():

#Partie pour demander à l'utilisateur d'entrer les valeurs nécessaires     
    u = float(input("Entrez la valeur initiale de u  :  "))
    E = float(input("Entrez la valeur du seuil de précision :  "))
    S = u;
    print("\nInitialement on a :  u = ",u," et S = ", S)
#On écris maintenant des fonctions pour demander à l'utilisateur d'entrer l'expression de sa fonction
    def Entrer_fonction(): 
        F = input("Entrez votre fonction en fonction de u (exemple: u**2 - 6u + 4):  ")
        return lambda u: eval(F)

# De même, on demande à l'ulisateur d'entrer l'expression de la dérrivée de sa fonction 
    def Entrer_der():
        Der_F = input("Entrez la dérrivée de votre fonction  :  ")
        return lambda u: eval(Der_F)
            
#Ici on affecte les valeurs de la fonction puis celle de sa dérrivée correspondente respectivement à F et Der_F entrées par l'utilisateur
    F = Entrer_fonction()
    Der_F = Entrer_der()
    
            
    if (Der_F(u) != 0 ) :
        S = u - (F(u)/Der_F(u)) # calcul du terme suivant de U
    
    print("\n En suite S = ", S)
    while ( (Der_F(S) != 0) and (abs(S-u) >= E ) ): # La structure tantque pour faire les test sur la dérrivée de f et la velur absolue de (s-u)
        print("Der_F(",S,") = ", Der_F(S), " et |S-u| = ", abs(S-u) )

        u = S
        print("u = ", u)
        print("Calcul de la nouvelle valeur de S")
        S =u - (F(u)/Der_F(u))
        print("la nouvelle valeur de S = ", S,"\n")
        
        
    print(" \nOn a en fin   :  u = ", u, ";   S = ", S, "; avec Der_F(",S,") = ", Der_F(S), "  et  |S-u| = ", abs(S-u))
    
    return print( " \n\n\n            La valeur approchée de votre fonction à ",E," près est : S = ", S); # Retour du résultat final 

if __name__ == "main" :
    print(Newton())# Affichage final

















def Secante(): 

    print("Soit votre fonction définie sur l'intervalle [a,b]");

    a = float(input("Entrez la valeur de a :  ")) ;
    b = float(input("Entrez la valeur de b :  ")) ; 
    e = float(input("Entrez la valeur du seuil de précision :  ")) ;
    S = a ;



    def donne_fonc() :
        W = input(" Veuillez entrer votre fonction  f(x) ");
        return lambda x: eval(W)


    f = donne_fonc() ;

    if f(b) == 0 :
        S = b ;

    print("Les valeurs initialles sont :  a = ", a ," ;  b = ", b , " ; e = ", e , " ; S = ", S )
    
    while ( (f(a)*f(b) < 0 )  and (abs(b-a) >= e)  ) :
        print("\nf(",a,") * f(",b,") = ", (f(a)*f(b)) ," et  |",a,"- ",b,"| = ", abs(b-a)) ;
        
        S = (a*f(b) - b*f(a))/(f(b) - f(a)) ;
        
        print("Donc le nouveau  S = ", S)
        
        if (f(S)*f(b)) < 0 :
            a = S ;
            print("Comme (f(S)*f(b)) = ",(f(S)*f(b)) ,"< 0  alors a = ", S)
        else :
            b = S ;
            print("Comme (f(S)*f(b)) = ",(f(S)*f(b)) ," > 0  alors b = ", S)
        
    

        
    return print("\nf(",a,") * f(",b,") = ", (f(a)*f(b)) ," et  |",a,"- ",b,"| = ", abs(b-a),"\n alors on a finalement :  a = ",a," ; b = ",b," et  S = ",S,"\n\nLa solution approchée de votre fonction à ",e," près est  : ", S)
        
if __name__ == "__main__" :
    print(Secante())

















import numpy as np

def pivot_de_gauss():
    """
    Effectue la méthode du pivot de Gauss complète :
    - saisie de la matrice augmentée
    - réduction échelonnée
    - détection des types de solutions
    - affichage des résultats
    """

    print("=== Méthode du Pivot de Gauss (tout-en-un) ===")
    n = int(input("Entrez le nombre d'équations (et d'inconnues) : "))

    print("\nEntrez la matrice augmentée (coefficients + termes constants) :")
    print(f"(Chaque ligne doit contenir {n+1} nombres séparés par des espaces)")

    # Saisie de la matrice augmentée
    A = []
    for i in range(n):
        ligne = list(map(float, input(f"Ligne {i+1} : ").split()))
        if len(ligne) != n + 1:
            raise ValueError(f" La ligne doit contenir exactement {n+1} valeurs.")
        A.append(ligne)

    A = np.array(A, dtype=float)

    print("\n=== Matrice augmentée initiale ===")
    print(A)

    # Étape 1 : Réduction de Gauss
    n, m = A.shape
    for i in range(min(n, m-1)):
        # Trouver le pivot maximal
        pivot = i + np.argmax(np.abs(A[i:, i]))
        if abs(A[pivot, i]) < 1e-12:
            continue

        # Échange de lignes
        if pivot != i:
            A[[i, pivot]] = A[[pivot, i]]

        # Normalisation du pivot
        A[i] = A[i] / A[i, i]

        # Élimination dans les autres lignes
        for k in range(n):
            if k != i:
                A[k] -= A[k, i] * A[i]

    print("\n=== Matrice échelonnée réduite ===")
    print(A)

    # Étape 2 : Analyse du système
    tol = 1e-10
    coeff = A[:, :-1]
    rang_A = np.linalg.matrix_rank(coeff, tol)
    rang_AA = np.linalg.matrix_rank(A, tol)

    # Détection des lignes contradictoires
    for ligne in A:
        if np.all(np.abs(ligne[:-1]) < tol) and abs(ligne[-1]) > tol:
            print("\n Le système est incompatible : aucune solution.")
            return

    if rang_A < rang_AA:
        print("\n Le système est incompatible : aucune solution.")
    elif rang_A < m - 1:
        print("\n Le système admet une infinité de solutions (paramétriques possibles).")
    else:
        solutions = A[:, -1]
        print("\n Système à solution unique :")
        for i, val in enumerate(solutions, start=1):
            print(f"x{i} = {val:.4f}")



#  Exécution du programme

if __name__=="main" :
    pivot_de_gauss()
