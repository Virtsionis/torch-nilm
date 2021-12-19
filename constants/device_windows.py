from constants.enumerates import ElectricalAppliances


WINDOWS = {
    'VIBFNET': {
        ElectricalAppliances.MICROWAVE: 50,
        ElectricalAppliances.KETTLE: 50,
        ElectricalAppliances.FRIDGE: 350,
        ElectricalAppliances.WASHING_MACHINE: 150,
        ElectricalAppliances.DISH_WASHER: 450,
    },
    'FNET': {
        ElectricalAppliances.MICROWAVE: 50,
        ElectricalAppliances.KETTLE: 50,
        ElectricalAppliances.FRIDGE: 350,
        ElectricalAppliances.WASHING_MACHINE: 150,
        ElectricalAppliances.DISH_WASHER: 450,
    },
    'S2P': {
        ElectricalAppliances.MICROWAVE: 100,
        ElectricalAppliances.KETTLE: 300,
        ElectricalAppliances.FRIDGE: 400,
        ElectricalAppliances.WASHING_MACHINE: 400,
        ElectricalAppliances.DISH_WASHER: 500,
    },
    'WGRU': {
        ElectricalAppliances.MICROWAVE: 100,
        ElectricalAppliances.KETTLE: 150,
        ElectricalAppliances.FRIDGE: 450,
        ElectricalAppliances.WASHING_MACHINE: 150,
        ElectricalAppliances.DISH_WASHER: 350,
    },
    'SAED': {
        ElectricalAppliances.MICROWAVE: 100,
        ElectricalAppliances.KETTLE: 50,
        ElectricalAppliances.FRIDGE: 250,
        ElectricalAppliances.WASHING_MACHINE: 50,
        ElectricalAppliances.DISH_WASHER: 200,
    },
    'SimpleGru': {
        ElectricalAppliances.MICROWAVE: 100,
        ElectricalAppliances.KETTLE: 200,
        ElectricalAppliances.FRIDGE: 400,
        ElectricalAppliances.WASHING_MACHINE: 200,
        ElectricalAppliances.DISH_WASHER: 450,
    },
}
