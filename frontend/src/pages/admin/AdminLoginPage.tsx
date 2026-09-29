import { useState } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { Alert, Button, Card, Form, Input, Typography } from 'antd'
import {
  FundOutlined,
  LockOutlined,
  LoginOutlined,
  SafetyCertificateOutlined,
  UserOutlined,
} from '@ant-design/icons'
import { adminLogin } from '@/api/admin'
import { useAppStore } from '@/stores/useAppStore'
import './admin-login.css'

const { Title, Text } = Typography

interface LoginValues {
  username: string
  password: string
}

interface LoginLocationState {
  from?: string
}

export default function AdminLoginPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const isAuthenticated = useAppStore((s) => s.isAdminAuthenticated)
  const login = useAppStore((s) => s.login)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [capsLock, setCapsLock] = useState(false)

  if (isAuthenticated) {
    return <Navigate to="/admin/monitor" replace />
  }

  const handleSubmit = async (values: LoginValues) => {
    setLoading(true)
    setError(null)
    try {
      const response = await adminLogin(values.username.trim(), values.password)
      if (!response?.success || !response.token) {
        setError(response?.message || '登录失败，请核对账号和密码')
        return
      }

      login(response.token)
      const state = location.state as LoginLocationState | null
      const destination = state?.from?.startsWith('/admin/')
        ? state.from
        : '/admin/monitor'
      navigate(destination, { replace: true })
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="admin-login-shell">
      <section className="admin-login-intro" aria-label="Kronos 管理中心">
        <div className="admin-login-brand">
          <span className="admin-login-logo"><FundOutlined /></span>
          <span>
            <strong>Kronos</strong>
            <small>系统运维后台</small>
          </span>
        </div>
        <div className="admin-login-intro-copy">
          <SafetyCertificateOutlined />
          <Title level={1}>管理中心</Title>
          <Text>模型运行、数据状态与系统配置统一管理</Text>
        </div>
        <Text className="admin-login-version">Kronos Strategy System</Text>
      </section>

      <section className="admin-login-form-area">
        <Card className="admin-login-card" bordered>
          <header className="admin-login-card-header">
            <Title level={3}>管理员登录</Title>
            <Text type="secondary">使用管理员凭据验证身份</Text>
          </header>

          {error && (
            <Alert
              message={error}
              type="error"
              showIcon
              closable
              onClose={() => setError(null)}
              className="admin-login-alert"
            />
          )}

          <Form<LoginValues>
            layout="vertical"
            initialValues={{ username: 'admin' }}
            onFinish={handleSubmit}
            requiredMark={false}
            size="large"
          >
            <Form.Item
              name="username"
              label="管理员账号"
              rules={[
                { required: true, message: '请输入管理员账号' },
                { whitespace: true, message: '管理员账号不能为空' },
              ]}
            >
              <Input
                prefix={<UserOutlined />}
                placeholder="管理员账号"
                autoComplete="username"
                autoFocus
              />
            </Form.Item>

            <Form.Item
              name="password"
              label="管理员密码"
              extra={capsLock ? '大写锁定已开启' : undefined}
              rules={[
                { required: true, message: '请输入管理员密码' },
                { min: 6, message: '密码至少为 6 位' },
              ]}
            >
              <Input.Password
                prefix={<LockOutlined />}
                placeholder="管理员密码"
                autoComplete="current-password"
                onKeyUp={(event) => setCapsLock(event.getModifierState('CapsLock'))}
                onBlur={() => setCapsLock(false)}
              />
            </Form.Item>

            <Button
              type="primary"
              htmlType="submit"
              icon={<LoginOutlined />}
              loading={loading}
              block
              className="admin-login-submit"
            >
              登录管理中心
            </Button>
          </Form>

          <div className="admin-login-notice">
            <SafetyCertificateOutlined />
            <Text type="secondary">仅限授权管理员访问</Text>
          </div>
        </Card>
      </section>
    </main>
  )
}
