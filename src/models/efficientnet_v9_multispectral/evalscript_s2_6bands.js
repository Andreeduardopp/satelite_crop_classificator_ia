//VERSION=3
//
// Sentinel-2 L2A — 6 bandas para classificação de culturas (V9).
//
// Bandas selecionadas (resolução nativa):
//   B02  Blue       490 nm   10 m
//   B03  Green      560 nm   10 m
//   B04  Red        665 nm   10 m
//   B08  NIR        842 nm   10 m   <- saúde da vegetação
//   B11  SWIR1     1610 nm   20 m   <- umidade do dossel
//   B12  SWIR2     2190 nm   20 m   <- umidade / lignina
//
// Output: FLOAT32, valores de reflectância em [0, 1]. O Sentinel Hub
// reamostra B11/B12 para a resolução do bbox (10 m) no servidor.
//
// Esta layer é o substituto direto da BANDAS_RBN do dashboard. Para usar
// no Processing API basta enviar este evalscript junto com o request —
// nenhuma layer precisa estar pre-configurada.

function setup() {
    return {
        input: ["B02", "B03", "B04", "B08", "B11", "B12"],
        output: {
            bands: 6,
            sampleType: "FLOAT32"
        }
    };
}

function evaluatePixel(sample) {
    return [
        sample.B02,
        sample.B03,
        sample.B04,
        sample.B08,
        sample.B11,
        sample.B12
    ];
}
