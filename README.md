# MOL O2C Lead Time Analytics (helyi projekt)
1. `pip install -r requirements.txt`
2. Tedd az Excelt ide: `data/Lead_Time_between_steps_in_hour.xlsx`
3. VS Code: Terminal > Run Task > "1) Adat feldolgozása", majd "2) Dashboard indítása" -> http://localhost:5500
(vagy: `python build_data.py` és `python -m http.server 5500`)
Az `index.html` a `data/orders.json`-t olvassa; Excel-frissítéskor csak az 1. lépést kell újra futtatni.

## ML-előkészítés (backend, még nincs ML)
`pip install -r requirements.txt -r backend/requirements-backend.txt`, majd 1) task (létrehozza a `data/processed/` mappát), utána 3) task -> http://localhost:8000/docs.
Részletek: `docs/ML_ARCHITECTURE.md`. A dashboard (`index.html`) érintetlen, a backend nélkül is működik.

## Machine Learning fül
1) task -> 2) task (dashboard, :5500) és 3) task (backend, :8000) egyszerre fut. A fülön élő: EDA, Korrelációelemzés, Lineáris, Többváltozós lineáris, Random Forest regresszió, Változófontosság és modellértelmezés, Átfutási idő előrejelzése, Modellértékelés és összehasonlítás – mind a 8 alfül kész. Új csomagok miatt futtasd újra: `pip install -r backend/requirements-backend.txt`. Ha a backend nem fut, a fül jelzi, a másik két oldal így is működik.

> Python 3.9-től működik (a backend Python 3.9-en és 3.12-n is tesztelve).

Új csomag (SHAP): `pip install -r backend/requirements-backend.txt` — nélküle is működik az oldal, csak a SHAP rész hiányzik.

## Export fül (PDF riport)
Az **Export** fülön kiválaszthatod a riport tartalmát (az Áttekintés mindig benne van; a Medián átfutási idő és a Machine Learning opcionális, az ML alfülei külön jelölhetők), majd „Riport készítése" → előnézet → „PDF letöltése". A riport tartalmazza az aktuális szűrőket és beállításokat. A közvetlen letöltéshez a gépen telepített Chrome / Edge szükséges (a backend használja); ha nincs, a böngésző nyomtatási ablaka nyílik meg – válaszd a „Mentés PDF-ként" lehetőséget. Az ML oldalakhoz fusson a backend.

## Egyszerűbb indítás: egyetlen terminál
A backend a dashboardot is kiszolgálja. Elég a backend elindítása (`cd backend`, majd `python -m uvicorn app.main:app --port 8000`), utána nyisd meg: **http://localhost:8000**. A `python -m http.server 5500` nem kell hozzá. Így a böngésző nem blokkolja a kéréseket, és a Machine Learning fül, valamint az Export is azonnal működik. A régi mód (5500-as szerver + 8000-es backend) is megy, ha a dashboardot `http://localhost:5500` vagy `http://127.0.0.1:5500` címen nyitod meg.
