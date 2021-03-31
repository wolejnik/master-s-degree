import folium
from folium.plugins import TimeSliderChoropleth, TimestampedGeoJson
import geopandas as gpd
import pandas as pd
import branca.colormap as cm
import warnings
import numpy as np
from shapely import geometry
import matplotlib as mpl
import matplotlib.cm as cm1
from matplotlib.colors import to_hex

warnings.filterwarnings("ignore")


def zad1():
    gdf = gpd.read_file("folium/country.geojson")
    covid = pd.read_csv('folium/owid-covid-data.csv')
    m = folium.Map(
        location=[53.4327, 14.5483],
        zoom_start=2,
        tiles='OpenStreetMap',
        crs='EPSG3857')
    folium.GeoJson(
            gdf,
            name="Państwa",
            smooth_factor=2
        ).add_to(m)
    m.save(outfile='zad1.html')


def zad2():
    gdf = gpd.read_file("folium/country.geojson")

    covid = pd.read_csv('folium/owid-covid-data.csv')
    covid = covid[~covid.iso_code.str.startswith('OWID')]
    new_covid = gpd.GeoDataFrame(columns=['iso_code', 'date', 'total_cases_per_million'])
    i = 0
    for country in covid["iso_code"].unique():
        most_recent_date = max(covid[covid['iso_code'] == country]['date'])
        new_covid.loc[i] = [country,
                            most_recent_date,
                            covid[(covid['iso_code'] == country) & (covid['date'] == most_recent_date)]['total_cases_per_million'].values[0]]
        i += 1

    gdf_merged = gdf.merge(new_covid, left_on='GMI_CNTRY', right_on='iso_code')

    m = folium.Map(
        location=[53.4327, 14.5483],
        zoom_start=2,
        tiles='OpenStreetMap',
        crs='EPSG3857')

    bins = list(new_covid['total_cases_per_million'].quantile([0, 0.4, 0.6, 0.8, 0.9, 1]))
    print(bins)
    bins1 = list(new_covid['total_cases_per_million'].quantile([x/10 for x in range(0, 11, 2)]))
    print(bins1)

    choropleth = folium.Choropleth(
        geo_data=gdf_merged,
        name='Liczba zachorowań',
        data=new_covid,
        columns=['iso_code', 'total_cases_per_million'],
        key_on='feature.properties.GMI_CNTRY',
        fill_color='OrRd',
        fill_opacity=0.7,
        line_opacity=0.1,
        legend_name='Liczba zachorowań',
        bins=bins,
        reset=True,
        highlight=True,
    ).add_to(m)
    test = folium.plugins.TimestampedGeoJson(None)
    test.add_child(folium.GeoJsonTooltip(
        fields=['total_cases_per_million']
    ))
    folium.GeoJson(
        gdf_merged,
        name="Państwa",
        smooth_factor=2,
        style_function=lambda x: {'color': 'black', 'fillColor': 'transparent', 'weight': 2},
        tooltip=folium.GeoJsonTooltip(fields=['total_cases_per_million'], labels=False, sticky=False),
        highlight_function=lambda x: {'weight': 3, 'fillColor': 'gray'}
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(outfile='zad2.html')


def zad3():
    gdf = gpd.read_file("folium/country.geojson")
    covid = pd.read_csv('folium/owid-covid-data.csv')
    covid = covid[~covid.iso_code.str.startswith('OWID')]
    covid['total_cases_log'] = np.log10(covid['total_cases'])
    covid = covid[covid['date'] <= '2020-05-04']

    first_date = min(covid['date'])
    last_date = max(covid['date'])
    date_range = pd.date_range(start=first_date, end=last_date)
    dt_index_epochs = date_range.astype(int) // 10 ** 9
    dt_index = dt_index_epochs.astype("U10")

    styledata = {}
    j = 0
    for country in gdf['GMI_CNTRY']:
        name = gdf[gdf['GMI_CNTRY'] == country]['CNTRY_NAME'].values[0]
        df = pd.DataFrame(
            {
                'name': name,
                'color': 0,
                'opacity': 0.5,
                'cases': 0,
                'daily': 0,
            },
            index=dt_index)
        for i, dt in enumerate(dt_index):
            df.at[dt, 'name'] = name
            df.at[dt, 'opacity'] = 0.5
            if covid[(covid['iso_code'] == country) & (covid['date'] == str(date_range[i].date()))]["total_cases_log"].values > 0:
                df.at[dt, 'color'] = covid[(covid['iso_code'] == country) & (covid['date'] == str(date_range[i].date()))]["total_cases_log"].values[0]
                df.at[dt, 'cases'] = covid[(covid['iso_code'] == country) & (covid['date'] == str(date_range[i].date()))]["total_cases"].values[0]
            else:
                df.at[dt, 'color'] = 0
                df.at[dt, 'cases'] = 0
            if covid[(covid['iso_code'] == country) & (covid['date'] == str(date_range[i].date()))]["new_cases"].values > 0:
                df.at[dt, 'daily'] = covid[(covid['iso_code'] == country) & (covid['date'] == str(date_range[i].date()))]["new_cases"].values[0]
            else:
                df.at[dt, 'daily'] = 0
        styledata[j] = df.sort_index()
        j += 1


    max_color, min_color, max_opacity, min_opacity = 0, 0, 0, 0

    for country, data in styledata.items():
        max_color = max(max_color, data["color"].max())
        min_color = min(max_color, data["color"].min())

    cmap = cm.linear.YlOrRd_09.scale(min_color, max_color)

    for country, data in styledata.items():
        data["color"] = data["color"].apply(cmap)
        data["opacity"] = 0.5

    styledict = {
        str(country): data.to_dict(orient="index") for country, data in styledata.items()
    }

    m = folium.Map([53.4327, 14.5483],
                   tiles="OpenStreetMap",
                   max_bounds=True,
                   zoom_start=2,
                   )

    g = TimeSliderChoropleth(
        gdf.to_json(),
        styledict=styledict,
        name='Timeseries'
    ).add_to(m)

    _ = cmap.add_to(m)
    cmap.caption = "Log of total cases of Covid"
    folium.LayerControl().add_to(m)
    m.save(outfile='zad3.html')


if __name__ == '__main__':
    # zad1()
    # zad2()
    zad3()