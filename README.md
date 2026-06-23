# Life-Expectancy-Prediction-SAUuSAU
Predmetni projekat iz Softverskih algoritama u sistemima automatskog upravljanja - Predikcija životnog veka.

Student: Luka Lepojević

Broj indeksa: RA 160/2022

Studijski program: Računarstvo i automatika/Računarski upravljački sistemi

Godina studija: Treća godina

1. Opis korišćenog skupa podataka

Korišćeni skup podataka (dataset) preuzet je iz zvanične Global Health Observatory (GHO) baze podataka pod okriljem Svetske zdravstvene organizacije (WHO), a dopunjen je i ekonomskim podacima Ujedinjenih nacija.

    Vremenski i geografski obuhvat: Dataset pokriva istorijski period od 2000. do 2015. godine i obuhvata uzorke za 193 zemlje sveta.

    Struktura: Skup se sastoji od ukupno 2938 uzoraka (redova) raspoređenih kroz 22 atributa (kolone).

    Priroda promenljivih: Atributi su pažljivo odabrani tako da reprezentuju četiri ključna stuba razvoja jednog društva:

        Demografski pokazatelji: Smrtnost odraslih (Adult Mortality), smrtnost novorođenčadi (infant deaths), smrtnost dece do 5 godina (under-five deaths).

        Socio-ekonomski faktori: Indeks kompozicije resursa/ljudskog razvoja (Income composition of resources), stepen obrazovanja izražen kroz prosečan broj godina školovanja (Schooling), bruto domaći proizvod (GDP), ukupna potrošnja na zdravstvenu zaštitu (Total expenditure).

        Zdravstveni pokazatelji i imunizacija: Stopa vakcinisanosti protiv Hepatitisa B (Hepatitis B), Polio virusa (Polio) i Difterije (Diphtheria), rasprostranjenost HIV/AIDS-a, indeks telesne mase (BMI), konzumacija alkohola (Alcohol).

        Kategorijski status: Kategorizacija zemlje na razvijene (Developed) i zemlje u razvoju (Developing).

    Ciljna promenljiva (Target): Očekivani životni vek stanovništva (Life expectancy) izražen u godinama kao neprekidna numerička vrednost.

2. Objašnjenje izvršene obrade podataka (Preprocesiranje)

Pre nego što su podaci prosleđeni algoritmima mašinskog učenja, izvršen je rigorozan inženjerski proces čišćenja, filtriranja i transformacije u Fazi 1:

    Sintaksno čišćenje niza: Uočen je tehnički nedostatak u sirovom datasetu gde su nazivi kolona sadržali skrivene prazne razmake (engl. leading/trailing spaces, npr. 'Life expectancy '). Ovi razmaci su uspešno uklonjeni primenom metode .str.strip().

    Filtriranje ciljne promenljive: Iz skupa podataka su momentalno uklonjeni svi redovi u kojima je nedostajala vrednost za Life expectancy, jer nadgledano mašinsko učenje (Supervised Learning) striktno zahteva postojanje tačne labele za obučavanje i validaciju.

    Napredna grupna imputacija: Umesto proste zamene nedostajućih vrednosti (NaN) globalnim prosekom cele baze (što bi narušilo specifičnosti različitih ekonomija), implementirana je grupisana imputacija po državama. Nedostajuće vrednosti za određeni atribut (npr. GDP) u nekoj godini popunjene su aritmetičkom sredinom tog istog atributa isključivo za tu konkretnu državu kroz ostale godine. Ukoliko neka država uopšte nije imala podatke za dati atribut, primenjen je globalni prosek tog atributa kao sekundarni korak.

    Enkodiranje kategorijskih promenljivih: Atribut Status (tekstualni podatak "Developed" / "Developing") uspešno je mapiran u binarni numerički format, gde je vrednost 1 dodeljena razvijenim zemljama, a 0 zemljama u razvoju.

    Podela i skaliranje: Podaci su podeljeni u razmeri 80% za obučavanje (trening skup) i 20% za testiranje (test skup). Izvršena je standardizacija (skaliranje) pomoću StandardScaler-a kako kolone sa velikim rasponima brojeva (poput GDP-a ili Population-a) ne bi veštački nadjačale ostale atribute u linearnim modelima.

3. Prikaz najvažnijih rezultata eksplorativne analize (EDA)

Eksplorativna analiza podataka u Fazi 2 rezultovala je donošenjem ključnih odluka o arhitekturi modela kroz generisanje grafičkih izveštaja:

    Raspodela targeta (1_raspodela_targeta.png): Histogram je pokazao blago levo-asimetričnu raspodelu. Najveća učestalost uzoraka je u opsegu od 70 do 75 godina, sa dugim "repom" ka nižim vrednostima koji pripada siromašnijim regionima.

    Matrica korelacije i multikolinearnost (2_matrica_korelacije.png): Izračunavanjem Pearsonovih koeficijenata detektovan je ozbiljan problem savršene multikolinearnosti. Atributi infant deaths i under-five deaths imaju korelaciju 1.00, dok GDP i percentage expenditure imaju korelaciju 0.90.

    Obrada i filtriranje na osnovu EDA analize: Kako multikolinearnost dramatično destabilizuje linearne modele, doneta je opravdana odluka o trajnom uklanjanju redundantnih kolona: infant deaths, percentage expenditure, thinness 5-9 years, kao i kolone Country (tekstualni ključ) i Population (koja ima korelaciju blisku nuli sa targetom). Skup je bezbedno redukovan sa 21 na 16 stabilnih atributa.

    Detekcija anomalija (5_detekcija_anomalija.png): Boxplot dijagrami su otkrili prisustvo velikog broja ekstremnih vrednosti (autlajera) u kolonama HIV/AIDS i Adult Mortality. Budući da ovi podaci predstavljaju realne istorijske epidemiološke krize, doneta je odluka da se oni ne brišu, već da se u sledećoj fazi primene napredni nelinearni algoritmi koji su prirodno otporni na autlajere.

4. Opis korišćenih modela

U skladu sa regresionom prirodom problema, u Fazi 3 i Fazi 4 implementirana su i testirana tri različita matematička algoritma:

    Linearna regresija (Linear Regression): Parametarski model koji pokušava da uspostavi linearnu hiperravan između ulaznih atributa i očekivanog životnog veka. Služi kao osnovni model (Baseline).

    Gradient Boosting Regressor: Ansambl model koji koristi tehniku pojačavanja (Boosting). On sekvencijalno gradi plitka stabla odlučivanja, gde svako naredno stablo minimizuje grešku (reziduale) prethodnog kroz gradijentni pad.

    Random Forest Regressor: Moćni neparametarski ansambl model zasnovan na tehnici pakovanja (Bagging). Algoritam kreira veliki broj nezavisnih stabala odlučivanja (u našem slučaju do 200) nad nasumičnim podskupovima podataka i atributa, a krajnju predikciju donosi agregacijom (srednjom vrednošću) izlaza svih pojedinačnih stabala. Izuzetno je otporan na preprilagođavanje (overfitting) i autlajere zabeležene u EDA fazi.

Nad Random Forest modelom je u Fazi 4 izvršeno i fino podešavanje (Tuning) parametara kroz GridSearchCV unakrsnu validaciju sa 3 nabora.

5. Prikaz performansi modela

Kvalitet modela evaluiran je kroz tri standardne metrike mašinskog učenja: MAE (Srednja apsolutna greška), RMSE (Kvadratna greška) i R2 Score (Koeficijent determinacije).

Tabela 1: Uporedni prikaz polaznih (Baseline) modela:
Algoritam	            MAE (godine)	RMSE	      R^2 Score
Linear Regression	    2.802155	    3.693404	  0.842351
Gradient Boosting	    1.254102	    1.832210	  0.961204
Random Forest (Base)	1.043210	    1.662144	  0.968112

Nakon sprovođenja GridSearchCV optimizacije, pronađeni su najbolji hiperparametri: n_estimators=200, max_depth=20, min_samples_split=2, max_features=None.

Konačne performanse optimizovanog Random Forest modela (sa svih 16 selektovanih atributa):

    MAE: 1.026655 (Prosečna greška modela je svedena na svega 1 godinu i 9 dana)

    RMSE: 1.638804

    R^2 Score: 0.968951 (Model objašnjava čak 96.89% varijanse test podataka, što predstavlja izuzetan rezultat)

6. Analiza atributa koji najviše utiču na predikciju

Primenom ugrađenog mehanizma Gini značajnosti u optimizovanom Random Forest modelu (best_rf.feature_importances_), izvršeno je matematičko rangiranje uticaja atributa na donošenje odluka (6_znacajnost_atributa.png).

Tabela 2: Rang lista značajnosti atributa (Top 5):

    HIV/AIDS: 0.596860 (Koeficijent značajnosti od skoro 59.7%)

    Income composition of resources: 0.157122 (Značajnost oko 15.7%)

    Adult Mortality: 0.135464 (Značajnost oko 13.5%)

    Schooling: 0.021643 (Značajnost oko 2.1%)

    under-five deaths: 0.018863 (Značajnost oko 1.9%)

Svi preostali atributi pojedinačno (uključujući GDP, alkohol, BMI i vakcine) ostvaruju uticaj manji od 1.5%.

7. Tumačenje dobijenih rezultata

    Dominacija zdravstveno-ekonomskog trija: Rezultati jasno pokazuju da rasprostranjenost HIV/AIDS-a ima ubedljivo najveću težinu u globalnom skupu podataka za posmatrani istorijski period (2000–2015). Zajedno sa ekonomskim indeksom razvoja (Income composition) i opštom smrtnošću odraslih (Adult Mortality), ova tri faktora diktiraju preko 88% uspešnosti predikcije celokupnog modela.

    Analiza Inženjerskog kompromisa (Trade-off): Na osnovu rang liste, kreiran je redukovani model koji koristi samo Top 5 najvažnijih atributa. Rezultati poređenja su fascinantni:

Konfiguracija modela	              MAE (godine)	    RMSE	      R^2 Score
Kompletan model (16 atributa)	      1.026655	        1.638804	  0.968951
Redukovani model (Top 5 atributa)	  1.138966	        1.756666	  0.964325

Tumačenje: Smanjenjem broja ulaznih parametara za skoro 70% (sa 16 na 5), model je izgubio svega 0.004 (0.4%) na svom R^2 skoru, dok se prosečna greška povećala za zanemarljivih 40 dana. To dokazuje da preostalih 11 atributa unose veliku količinu informacionog šuma koji ansambl stabala uspešno neutrališe.

8. Zaključak o praktičnoj upotrebljivosti modela

Razvijeni redukovani model mašinskog učenja poseduje izuzetno visoku praktičnu upotrebljivost u realnom svetu iz sledećih razloga:

    Ekonomičnost prikupljanja podataka: Umesto kompleksnog i skupog administriranja 16 različitih zdravstvenih i statističkih parametara jedne države, krajnjem korisniku ili donosiocu zdravstvenih odluka dovoljno je da obezbedi svega 5 lako dostupnih parametara kako bi dobio predikciju vrhunske tačnosti (R^2=96.43%).

    Mogućnost brze integracije (Deployment): Uspešnom serijalizacijom modela u binarni fajl izabrani_model.pkl (Faza 5), model je osposobljen za trenutno izvršavanje u realnom vremenu unutar bilo koje veb platforme, mobilne aplikacije ili API servisa, što je i demonstrirano kroz stabilan interaktivni korisnički interfejs u terminalu (gde za unete stabilne parametre model uspešno i logično generiše predikciju od 84.84 godine života).

    Ograničenja modela: Kao jedino realno ograničenje navodi se činjenica da je model obučen na istorijskim podacima do 2015. godine. Za modernu upotrebu u 2026. godini, model bi bilo poželjno periodično ažurirati (engl. retrain) najnovijim podacima SZO kako bi se uzele u obzir nove globalne zdravstvene promene (poput post-pandemijskih statistika).
