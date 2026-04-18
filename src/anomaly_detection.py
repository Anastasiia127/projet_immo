import numpy as np
import pandas as pd
from scipy import stats


def calcular_diferencial(y_real, y_pred):
    """
    Calcula el diferencial absoluto y porcentual entre precio real y predicho.

    Parámetros
    ----------
    y_real : array-like — precios reales
    y_pred : array-like — precios predichos por el modelo

    Devuelve
    --------
    DataFrame con columnas: diferencial, diferencial_pct
    """
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)

    diferencial     = y_real - y_pred
    diferencial_pct = np.where(y_pred != 0, (diferencial / y_pred) * 100, np.nan)

    return pd.DataFrame({
        "diferencial":     diferencial,
        "diferencial_pct": diferencial_pct,
    })


def clasificar_zonas(df, col_zona="city", col_real="price", col_pred="price_pred", confianza=0.95):
    """
    Clasifica zonas geográficas como 'Tendencial' o 'Atípica' usando
    un intervalo de confianza sobre el diferencial porcentual.

    Parámetros
    ----------
    df         : DataFrame con los datos
    col_zona   : nombre de la columna de zona geográfica (ej. 'city')
    col_real   : nombre de la columna con el precio real
    col_pred   : nombre de la columna con el precio predicho
    confianza  : nivel de confianza del intervalo (por defecto 0.95 → IC 95%)

    Devuelve
    --------
    DataFrame con una fila por zona y columnas:
        zona, precio_real_mediano, precio_pred_mediano,
        diferencial_pct_mediano, clasificacion, n_viviendas

    Ejemplo de uso
    --------------
    from src.anomaly_detection import clasificar_zonas

    # df_test debe tener columnas: city, price, price_pred
    df_test['price_pred'] = modelo.predict(X_test)
    resultado = clasificar_zonas(df_test, col_zona='city', col_real='price', col_pred='price_pred')
    print(resultado)
    """
    df = df.copy()

    # Calcular diferencial por fila
    df["_dif_pct"] = (df[col_real] - df[col_pred]) / df[col_pred].replace(0, np.nan) * 100

    # Agrupar por zona
    zona_stats = (
        df.groupby(col_zona)
        .agg(
            precio_real_mediano=(col_real, "median"),
            precio_pred_mediano=(col_pred, "median"),
            diferencial_pct_mediano=("_dif_pct", "median"),
            n_viviendas=(col_real, "count"),
        )
        .reset_index()
        .rename(columns={col_zona: "zona"})
    )

    # Calcular límites del intervalo de confianza
    alpha   = 1 - confianza
    z_score = stats.norm.ppf(1 - alpha / 2)  # 1.96 para IC 95%

    media  = zona_stats["diferencial_pct_mediano"].mean()
    std    = zona_stats["diferencial_pct_mediano"].std()
    limite = z_score * std

    zona_stats["limite_inferior"] = media - limite
    zona_stats["limite_superior"] = media + limite

    # Clasificar
    zona_stats["clasificacion"] = zona_stats["diferencial_pct_mediano"].apply(
        lambda x: "Atípica" if (x < media - limite or x > media + limite) else "Tendencial"
    )

    return zona_stats.sort_values("diferencial_pct_mediano", key=abs, ascending=False)


def resumen_anomalias(zona_stats):
    """
    Imprime un resumen de las zonas atípicas detectadas.

    Parámetros
    ----------
    zona_stats : DataFrame devuelto por clasificar_zonas()
    """
    total      = len(zona_stats)
    atipicas   = zona_stats[zona_stats["clasificacion"] == "Atípica"]
    tendencial = zona_stats[zona_stats["clasificacion"] == "Tendencial"]

    print("=" * 50)
    print(f"RESUMEN DE ANOMALÍAS (IC {zona_stats['limite_superior'].iloc[0]:.1f}%)")
    print("=" * 50)
    print(f"  Zonas analizadas:    {total}")
    print(f"  Zonas tendenciales:  {len(tendencial)}")
    print(f"  Zonas atípicas:      {len(atipicas)}")
    print()

    if not atipicas.empty:
        print("Zonas atípicas (ordenadas por diferencial):")
        for _, row in atipicas.iterrows():
            signo = "↑" if row["diferencial_pct_mediano"] > 0 else "↓"
            print(f"  {signo} {row['zona']:<30} {row['diferencial_pct_mediano']:+.1f}%  ({int(row['n_viviendas'])} viviendas)")
    print("=" * 50)


def exportar_resultados(zona_stats, path="../outputs/predicciones.csv"):
    """
    Exporta los resultados de clasificación a CSV.

    Parámetros
    ----------
    zona_stats : DataFrame devuelto por clasificar_zonas()
    path       : ruta donde guardar el CSV
    """
    zona_stats.to_csv(path, index=False)
    print(f"Resultados exportados a: {path}")
