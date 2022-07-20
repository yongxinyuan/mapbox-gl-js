import { parseCSSColor } from "csscolorparser";

class Color {
    r: number;
    g: number;
    b: number;
    a: number;

    static black: Color;
    static white: Color;
    static transparent: Color;
    static red: Color;
    static blue: Color;

    constructor(r: number, g: number, b: number, a: number = 1) {
        this.r = r;
        this.g = g;
        this.b = b;
        this.a = a;
    }

    toArray(): [number, number, number, number] {
        const { r, g, b, a } = this;
        return a === 0 ? [0, 0, 0, 0] : [
            r * 255 / a,
            g * 255 / a,
            b * 255 / a,
            a
        ];
    }
}

Color.black = new Color(0, 0, 0, 1);
Color.white = new Color(1, 1, 1, 1);
Color.transparent = new Color(0, 0, 0, 0);
Color.red = new Color(1, 0, 0, 1);
Color.blue = new Color(0, 0, 1, 1);

export default Color;