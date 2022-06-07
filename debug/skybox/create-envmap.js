import { drawInCanvas } from "./drawInCanvas.js";

const PLANES_VERTEX = `
precision highp float;
precision highp int;

attribute vec3 aPosition;
attribute vec2 aUv;
attribute float aFaceIndex;

varying vec3 vOutputDirection;

vec3 getDirection(vec2 uv, float face) {

    uv = 2.0 * uv - 1.0;

    vec3 direction = vec3(uv, 1.0);

    if(face == 0.0) {

        direction = direction.zyx; // ( 1, v, u ) pos x

    } else if(face == 1.0) {

        direction = direction.xzy;
        direction.xz *= -1.0; // ( -u, 1, -v ) pos y

    } else if(face == 2.0) {

        direction.x *= -1.0; // ( -u, v, 1 ) pos z

    } else if(face == 3.0) {

        direction = direction.zyx;
        direction.xz *= -1.0; // ( -1, v, -u ) neg x

    } else if(face == 4.0) {

        direction = direction.xzy;
        direction.xy *= -1.0; // ( -u, -1, v ) neg y

    } else if(face == 5.0) {

        direction.z *= -1.0; // ( u, v, -1 ) neg z

    }

    return direction;

}

void main() {
    vOutputDirection = getDirection(aUv, aFaceIndex);
    gl_Position = vec4(aPosition, 1.0);

}

`;
const PLANES_FRAGMENT = `
precision highp float;
precision highp int;

uniform float uFlipEnvMap;
uniform samplerCube uEnvMap;

varying vec3 vOutputDirection;

void main() {

    gl_FragColor = textureCube(uEnvMap, vec3(uFlipEnvMap * vOutputDirection.x, vOutputDirection.yz));

}

`;
const BLUR_VERTEX = `
precision highp float;
precision highp int;

attribute vec3 aPosition;
attribute vec2 aUv;
attribute float aFaceIndex;

varying vec3 vOutputDirection;

vec3 getDirection(vec2 uv, float face) {

    uv = 2.0 * uv - 1.0;

    vec3 direction = vec3(uv, 1.0);

    if(face == 0.0) {

        direction = direction.zyx; // ( 1, v, u ) pos x

    } else if(face == 1.0) {

        direction = direction.xzy;
        direction.xz *= -1.0; // ( -u, 1, -v ) pos y

    } else if(face == 2.0) {

        direction.x *= -1.0; // ( -u, v, 1 ) pos z

    } else if(face == 3.0) {

        direction = direction.zyx;
        direction.xz *= -1.0; // ( -1, v, -u ) neg x

    } else if(face == 4.0) {

        direction = direction.xzy;
        direction.xy *= -1.0; // ( -u, -1, v ) neg y

    } else if(face == 5.0) {

        direction.z *= -1.0; // ( u, v, -1 ) neg z

    }

    return direction;

}

void main() {
    vOutputDirection = getDirection(aUv, aFaceIndex);
    gl_Position = vec4(aPosition, 1.0);

}

`;
const BLUR_FRAGMENT = `
precision highp float;
precision highp int;

uniform float uFlipEnvMap;
uniform samplerCube uEnvMap;

varying vec3 vOutputDirection;

void main() {

    gl_FragColor = textureCube(uEnvMap, vec3(uFlipEnvMap * vOutputDirection.x, vOutputDirection.yz));

}

`;

const LOD_MIN = 4;
const IMAGE_SIZE = 512;
const LOD_MAX = Math.floor(Math.log2(IMAGE_SIZE));
const CUBE_SIZE = Math.pow(2, LOD_MAX);
const EXTRA_LOD_SIGMA = [0.125, 0.215, 0.35, 0.446, 0.526, 0.582];

export default function createEnvMap(gl, images) {
    const envWidth = 3 * Math.max(CUBE_SIZE, 16 * 7);
    const envHeight = 4 * CUBE_SIZE - 32;

    const planes = createPlanes(LOD_MAX);

    // 创建立方体纹理
    const cubeTexture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, cubeTexture);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_CUBE_MAP_NEGATIVE_X, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, images[0]);
    gl.texImage2D(gl.TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, images[1]);
    gl.texImage2D(gl.TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, images[2]);
    gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, images[3]);
    gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_Y, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, images[4]);
    gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_Z, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, images[5]);
    gl.generateMipmap(gl.TEXTURE_CUBE_MAP);

    // 帧缓冲区纹理
    const envTexture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, envTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, envWidth, envHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    // 创建帧缓冲区
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, envTexture, 0);

    gl.enable(gl.SCISSOR_TEST);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // 绘制原始比例的纹理
    drawOriginalImage(gl, planes, cubeTexture, envWidth, envHeight);

    // 绘制模糊的纹理
    drawBlurImage(gl, planes, cubeTexture, envWidth, envHeight);

    // debug env map
    // const pixels = new Uint8Array(envWidth * envHeight * 4);
    // gl.readPixels(0, 0, envWidth, envHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels)
    // drawInCanvas(
    //     pixels,
    //     envWidth,
    //     envHeight
    // );

    gl.disable(gl.SCISSOR_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return envTexture;
}

export function createEnvMapFromMapbox(gl, cubeEnv) {
    const envWidth = 3 * Math.max(CUBE_SIZE, 16 * 7);
    const envHeight = 4 * CUBE_SIZE - 32;

    const planes = createPlanes(LOD_MAX);

    // 帧缓冲区纹理
    const envTexture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, envTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, envWidth, envHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    // 创建帧缓冲区
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, envTexture, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, cubeEnv);

    gl.enable(gl.SCISSOR_TEST);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // 绘制原始比例的纹理
    drawOriginalImage(gl, planes, cubeEnv, envWidth, envHeight);

    // 绘制模糊的纹理
    drawBlurImage(gl, planes, cubeEnv, envWidth, envHeight);

    // debug env map
    const pixels = new Uint8Array(envWidth * envHeight * 4);
    gl.readPixels(0, 0, envWidth, envHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels)
    drawInCanvas(
        pixels,
        envWidth,
        envHeight
    );

    gl.disable(gl.SCISSOR_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return envTexture;
}

function createPlanes(lodMax) {
    const lodPlanes = [];
    const sizeLods = [];
    const sigmas = [];
    const totalLods = lodMax - LOD_MIN + 1 + EXTRA_LOD_SIGMA.length;

    let lod = lodMax;

    for (let i = 0; i < totalLods; i++) {
        const sizeLod = Math.pow(2, lod);
        sizeLods.push(sizeLod);

        const sigma = i > lodMax - LOD_MIN
            ? EXTRA_LOD_SIGMA[i - lodMax + LOD_MIN - 1]
            : i === 0
                ? 0
                : 1.0 / sizeLod;
        sigmas.push(sigma);

        const texelSize = 1.0 / (sizeLod - 1);
        const min = -texelSize / 2;
        const max = 1 + texelSize / 2;
        const uv1 = [min, min, max, min, max, max, min, min, max, max, min, max];

        const cubeFaces = 6;
        const vertices = 6;
        const positionSize = 3;
        const uvSize = 2;
        const faceIndexSize = 1;

        const position = new Array(positionSize * vertices * cubeFaces);
        const uv = new Array(uvSize * vertices * cubeFaces);
        const faceIndex = new Array(faceIndexSize * vertices * cubeFaces);

        for (let face = 0; face < cubeFaces; face++) {
            const x = (face % 3) * 2 / 3 - 1;
            const y = face > 2 ? 0 : - 1;
            const coordinates = [
                x, y, 0,
                x + 2 / 3, y, 0,
                x + 2 / 3, y + 1, 0,
                x, y, 0,
                x + 2 / 3, y + 1, 0,
                x, y + 1, 0
            ];
            position.splice(
                positionSize * vertices * face,
                positionSize * vertices,
                ...coordinates
            );
            uv.splice(
                uvSize * vertices * face,
                uvSize * vertices,
                ...uv1
            );
            faceIndex.splice(
                faceIndexSize * vertices * face,
                faceIndexSize * vertices,
                face, face, face, face, face, face
            );
        }

        lodPlanes.push({
            aPosition: {
                value: new Float32Array(position),
                elementsByComponent: positionSize
            },
            aUv: {
                value: new Float32Array(uv),
                elementsByComponent: uvSize
            },
            aFaceIndex: {
                value: new Float32Array(faceIndex),
                elementsByComponent: faceIndexSize
            }
        });

        if (lod > LOD_MIN) {
            lod--;
        }
    }

    return {
        lodPlanes,
        sizeLods,
        sigmas
    }
}

function drawOriginalImage(
    gl,
    planes,
    cubeMap,
    width,
    height
) {
    const plane = planes.lodPlanes[0];

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    const program = gl.createProgram();

    gl.shaderSource(vertexShader, PLANES_VERTEX);
    gl.shaderSource(fragmentShader, PLANES_FRAGMENT);

    gl.compileShader(vertexShader);
    gl.compileShader(fragmentShader);

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);

    gl.linkProgram(program);

    gl.validateProgram(program);

    if (!gl.getProgramParameter(program, gl.VALIDATE_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program) || "Error happened.");
    }

    const attributes = {};
    const uniforms = {};

    attributes.aPosition = gl.getAttribLocation(program, "aPosition");
    const aPositionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, aPositionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, plane.aPosition.value, gl.STATIC_DRAW);

    attributes.aUv = gl.getAttribLocation(program, "aUv");
    const aUvBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, aUvBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, plane.aUv.value, gl.STATIC_DRAW);

    attributes.aFaceIndex = gl.getAttribLocation(program, "aFaceIndex");
    const aFaceIndexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, aFaceIndexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, plane.aFaceIndex.value, gl.STATIC_DRAW);

    uniforms.uFlipEnvMap = gl.getUniformLocation(program, "uFlipEnvMap");
    uniforms.uEnvMap = gl.getUniformLocation(program, "uEnvMap");

    gl.viewport(0, 0, CUBE_SIZE * 3, CUBE_SIZE * 2);
    gl.scissor(0, 0, CUBE_SIZE * 3, CUBE_SIZE * 2);

    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, aPositionBuffer);
    gl.vertexAttribPointer(
        attributes.aPosition,
        3,
        gl.FLOAT,
        false,
        plane.aPosition.elementsByComponent * 4,
        0
    );
    gl.enableVertexAttribArray(attributes.aPosition);

    gl.bindBuffer(gl.ARRAY_BUFFER, aUvBuffer);
    gl.vertexAttribPointer(
        attributes.aUv,
        2,
        gl.FLOAT,
        false,
        plane.aUv.elementsByComponent * 4,
        0
    );
    gl.enableVertexAttribArray(attributes.aUv);

    gl.bindBuffer(gl.ARRAY_BUFFER, aFaceIndexBuffer);
    gl.vertexAttribPointer(
        attributes.aFaceIndex,
        1,
        gl.FLOAT,
        false,
        plane.aFaceIndex.elementsByComponent * 4,
        0
    );
    gl.enableVertexAttribArray(attributes.aFaceIndex);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, cubeMap);
    gl.uniform1i(uniforms.uEnvMap, 1);

    gl.uniform1f(uniforms.uFlipEnvMap, -1.0);

    gl.drawArrays(gl.TRIANGLES, 0, 36);



}

function drawBlurImage(
    gl,
    planes,
    cubeMap,
    width,
    height
) {
    for (let i = 1, len = planes.lodPlanes.length; i < len; i++) {
        const plane = planes.lodPlanes[i];

        const vertexShader = gl.createShader(gl.VERTEX_SHADER);
        const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
        const program = gl.createProgram();

        gl.shaderSource(vertexShader, BLUR_VERTEX);
        gl.shaderSource(fragmentShader, BLUR_FRAGMENT);

        gl.compileShader(vertexShader);
        gl.compileShader(fragmentShader);

        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);

        gl.linkProgram(program);

        gl.validateProgram(program);

        if (!gl.getProgramParameter(program, gl.VALIDATE_STATUS)) {
            throw new Error(gl.getProgramInfoLog(program) || "Error happened.");
        }

        const attributes = {};
        const uniforms = {};

        attributes.aPosition = gl.getAttribLocation(program, "aPosition");
        const aPositionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, aPositionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, plane.aPosition.value, gl.STATIC_DRAW);

        attributes.aUv = gl.getAttribLocation(program, "aUv");
        const aUvBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, aUvBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, plane.aUv.value, gl.STATIC_DRAW);

        attributes.aFaceIndex = gl.getAttribLocation(program, "aFaceIndex");
        const aFaceIndexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, aFaceIndexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, plane.aFaceIndex.value, gl.STATIC_DRAW);

        uniforms.uFlipEnvMap = gl.getUniformLocation(program, "uFlipEnvMap");
        uniforms.uEnvMap = gl.getUniformLocation(program, "uEnvMap");

        const outputSize = planes.sizeLods[i];
        const x = 3 * outputSize * (i > LOD_MAX - LOD_MIN ? i - LOD_MAX + LOD_MIN : 0);
        const y = 4 * (CUBE_SIZE - outputSize);
        gl.viewport(x, y, outputSize * 3, outputSize * 2);
        gl.scissor(x, y, outputSize * 3, outputSize * 2);

        gl.useProgram(program);

        gl.bindBuffer(gl.ARRAY_BUFFER, aPositionBuffer);
        gl.vertexAttribPointer(
            attributes.aPosition,
            3,
            gl.FLOAT,
            false,
            plane.aPosition.elementsByComponent * 4,
            0
        );
        gl.enableVertexAttribArray(attributes.aPosition);

        gl.bindBuffer(gl.ARRAY_BUFFER, aUvBuffer);
        gl.vertexAttribPointer(
            attributes.aUv,
            2,
            gl.FLOAT,
            false,
            plane.aUv.elementsByComponent * 4,
            0
        );
        gl.enableVertexAttribArray(attributes.aUv);

        gl.bindBuffer(gl.ARRAY_BUFFER, aFaceIndexBuffer);
        gl.vertexAttribPointer(
            attributes.aFaceIndex,
            1,
            gl.FLOAT,
            false,
            plane.aFaceIndex.elementsByComponent * 4,
            0
        );
        gl.enableVertexAttribArray(attributes.aFaceIndex);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_CUBE_MAP, cubeMap);
        gl.uniform1i(uniforms.uEnvMap, 1);

        gl.uniform1f(uniforms.uFlipEnvMap, -1.0);

        gl.drawArrays(gl.TRIANGLES, 0, 36);
    }
}