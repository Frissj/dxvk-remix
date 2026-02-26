/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "camera.h"
#include <cmath>

namespace dxvk {

Vector3 Camera::GetDirection() const {
    Vector3 dir = m_lookat - m_eye;
    float len = std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    if (len > 0.0f) {
        return Vector3(dir.x / len, dir.y / len, dir.z / len);
    }
    return Vector3(0, 0, -1);
}

void Camera::Translate(Vector3 const& v) {
    m_eye.x += v.x; m_eye.y += v.y; m_eye.z += v.z;
    m_lookat.x += v.x; m_lookat.y += v.y; m_lookat.z += v.z;
    m_changed = true;
}

void Camera::Rotate(float yaw, float pitch, float roll) {
    // Implementation for camera rotation
    m_changed = true;
}

void Camera::Roll(float speed) {
    m_changed = true;
}

void Camera::Dolly(float factor) {
    Vector3 dir = Vector3(m_lookat.x - m_eye.x, m_lookat.y - m_eye.y, m_lookat.z - m_eye.z);
    m_eye.x += dir.x * factor;
    m_eye.y += dir.y * factor;
    m_eye.z += dir.z * factor;
    m_changed = true;
}

void Camera::Pan(Vector2 speed) {
    m_changed = true;
}

void Camera::Zoom(const float factor) {
    m_fovY *= factor;
    m_changed = true;
}

void Camera::ComputeBasis(Vector3& u, Vector3& v, Vector3& w) const {
    // w = normalize(eye - lookat)
    w = Vector3(m_eye.x - m_lookat.x, m_eye.y - m_lookat.y, m_eye.z - m_lookat.z);
    float wLen = std::sqrt(w.x * w.x + w.y * w.y + w.z * w.z);
    if (wLen > 0.0f) {
        w.x /= wLen; w.y /= wLen; w.z /= wLen;
    }

    // u = normalize(cross(up, w))
    u = Vector3(
        m_up.y * w.z - m_up.z * w.y,
        m_up.z * w.x - m_up.x * w.z,
        m_up.x * w.y - m_up.y * w.x
    );
    float uLen = std::sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
    if (uLen > 0.0f) {
        u.x /= uLen; u.y /= uLen; u.z /= uLen;
    }

    // v = cross(w, u)
    v = Vector3(
        w.y * u.z - w.z * u.y,
        w.z * u.x - w.x * u.z,
        w.x * u.y - w.y * u.x
    );
}

Matrix4 Camera::GetViewMatrix() const {
    // Column-vector convention matching sample: mul(V, pos) = V * pos
    // Row-major storage: data[row][col], translation in column 3
    // W = normalize(lookat - eye) = forward direction
    // View Z stores -W (camera looks along -Z in view space, OpenGL convention)
    Vector3 W = Vector3(m_lookat.x - m_eye.x, m_lookat.y - m_eye.y, m_lookat.z - m_eye.z);
    float wLen = std::sqrt(W.x * W.x + W.y * W.y + W.z * W.z);
    if (wLen > 0.0f) { W.x /= wLen; W.y /= wLen; W.z /= wLen; }

    // U = normalize(cross(W, up)) — right direction
    Vector3 U = Vector3(
        W.y * m_up.z - W.z * m_up.y,
        W.z * m_up.x - W.x * m_up.z,
        W.x * m_up.y - W.y * m_up.x
    );
    float uLen = std::sqrt(U.x * U.x + U.y * U.y + U.z * U.z);
    if (uLen > 0.0f) { U.x /= uLen; U.y /= uLen; U.z /= uLen; }

    // V = normalize(cross(U, W)) — true up
    Vector3 V = Vector3(
        U.y * W.z - U.z * W.y,
        U.z * W.x - U.x * W.z,
        U.x * W.y - U.y * W.x
    );
    float vLen = std::sqrt(V.x * V.x + V.y * V.y + V.z * V.z);
    if (vLen > 0.0f) { V.x /= vLen; V.y /= vLen; V.z /= vLen; }

    float dotUeye = U.x * m_eye.x + U.y * m_eye.y + U.z * m_eye.z;
    float dotVeye = V.x * m_eye.x + V.y * m_eye.y + V.z * m_eye.z;
    float dotWeye = W.x * m_eye.x + W.y * m_eye.y + W.z * m_eye.z;

    Matrix4 mat;
    // Row 0: U (right)
    mat.data[0][0] = U.x;  mat.data[0][1] = U.y;  mat.data[0][2] = U.z;  mat.data[0][3] = -dotUeye;
    // Row 1: V (up)
    mat.data[1][0] = V.x;  mat.data[1][1] = V.y;  mat.data[1][2] = V.z;  mat.data[1][3] = -dotVeye;
    // Row 2: -W (negate forward → camera looks along -Z in view space)
    mat.data[2][0] = -W.x; mat.data[2][1] = -W.y; mat.data[2][2] = -W.z; mat.data[2][3] = dotWeye;
    // Row 3: homogeneous
    mat.data[3][0] = 0.0f; mat.data[3][1] = 0.0f; mat.data[3][2] = 0.0f; mat.data[3][3] = 1.0f;

    return mat;
}

Matrix4 Camera::GetProjectionMatrix() const {
    // Column-vector convention matching sample: mul(P, v) = P * v
    // OpenGL-style: depth range [-1, 1], clip.w = -pz (visible objects have w > 0)
    float tanHalfFovy = std::tan(m_fovY * 0.5f * 3.14159265f / 180.0f);

    Matrix4 mat;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            mat.data[i][j] = 0.0f;

    mat.data[0][0] = 1.0f / (m_aspectRatio * tanHalfFovy);
    mat.data[1][1] = 1.0f / tanHalfFovy;
    mat.data[2][2] = (m_zFar + m_zNear) / (m_zNear - m_zFar);
    mat.data[2][3] = 2.0f * m_zFar * m_zNear / (m_zNear - m_zFar);
    mat.data[3][2] = -1.0f;

    return mat;
}

Matrix4 Camera::GetViewProjectionMatrix() const {
    // Simple matrix multiplication
    Matrix4 proj = GetProjectionMatrix();
    Matrix4 view = GetViewMatrix();
    Matrix4 result;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result.data[i][j] = 0.0f;
            for (int k = 0; k < 4; ++k) {
                result.data[i][j] += proj.data[i][k] * view.data[k][j];
            }
        }
    }

    return result;
}

void Camera::SetEye(Vector3 eye) {
    m_eye = eye;
    m_changed = true;
}

void Camera::SetLookat(Vector3 lookat) {
    m_lookat = lookat;
    m_changed = true;
}

void Camera::SetUp(Vector3 up) {
    m_up = up;
    m_changed = true;
}

void Camera::SetFovY(float fovy) {
    m_fovY = fovy;
    m_changed = true;
}

void Camera::SetAspectRatio(float ar) {
    m_aspectRatio = ar;
    m_changed = true;
}

void Camera::SetNear(float nearPlane) {
    m_zNear = nearPlane;
    m_changed = true;
}

void Camera::SetFar(float farPlane) {
    m_zFar = farPlane;
    m_changed = true;
}

} // namespace dxvk
