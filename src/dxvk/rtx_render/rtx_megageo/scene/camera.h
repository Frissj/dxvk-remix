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

#pragma once

#include "../../../util/util_math.h"
#include "../../../util/util_vector.h"
#include "../../../util/util_matrix.h"

namespace dxvk {

class Camera
{
public:
    bool HasChanged() const { return m_changed; }

    Vector3 GetDirection() const;

    void Translate(Vector3 const& v);
    void Rotate(float yaw, float pitch, float roll);
    void Roll(float speed);
    void Dolly(float factor);
    void Pan(Vector2 speed);
    void Zoom(const float factor);

    Vector3 GetEye() const { return m_eye; }
    Vector3 GetLookat() const { return m_lookat; }
    Vector3 GetUp() const { return m_up; }

    float GetFovY() const { return m_fovY; }
    float GetAspectRatio() const { return m_aspectRatio; }
    float GetZNear() const { return m_zNear; }
    float GetZFar() const { return m_zFar; }

    Matrix4 GetViewMatrix() const;
    Matrix4 GetProjectionMatrix() const;
    Matrix4 GetViewProjectionMatrix() const;

    void SetEye(Vector3 eye);
    void SetLookat(Vector3 lookat);
    void SetUp(Vector3 up);
    void SetFovY(float fovy);
    void SetAspectRatio(float ar);
    void SetNear(float nearPlane);
    void SetFar(float farPlane);

private:
    void ComputeBasis(Vector3& u, Vector3& v, Vector3& w) const;

    Vector3 m_eye = Vector3(1.f, 1.f, 1.f);
    Vector3 m_lookat = Vector3(0.f, 0.f, 0.f);
    Vector3 m_up = Vector3(0.f, 1.f, 0.f);

    float m_fovY = 35.f;
    float m_aspectRatio = 1.f;
    float m_zNear = 0.1f;
    float m_zFar = 100.f;

    bool m_changed = true;
};

} // namespace dxvk
