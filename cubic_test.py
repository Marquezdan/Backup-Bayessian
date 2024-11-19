import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from helper.util import plot_regression
from laplace import Laplace

# Configuração
n_epochs = 1000
torch.manual_seed(711)

# Função para gerar dados de uma função cúbica com pontos aleatórios no intervalo [-3, 3]
def get_cubic_example_with_random_spacing(n, sigma_noise):
    # Gerar pontos aleatórios no intervalo [-3, 3]
    X_train = -3 + 6 * torch.rand(n).reshape(-1, 1)  # Intervalo de tamanho 6, centrado em 0

    # Aplicar a função cúbica com ruído
    y_train = X_train**3 + sigma_noise * torch.randn(X_train.shape)

    # Dados de teste cobrem o intervalo [-4, 4] para observar a continuidade da curva
    X_test = torch.linspace(-4, 4, 100).reshape(-1, 1)

    # Criar o DataLoader
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=len(X_train), shuffle=True)
    return X_train, y_train, train_loader, X_test

# Criar dados de regressão cúbicos com pontos aleatórios e intervalo [-3, 3]
X_train, y_train, train_loader, X_test = get_cubic_example_with_random_spacing(n=20, sigma_noise=0.3)

# Criar e treinar o modelo MAP
def get_model():
    torch.manual_seed(711)
    return torch.nn.Sequential(
        torch.nn.Linear(1, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 1)
    )

model = get_model()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(n_epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

# Aplicar a aproximação de Laplace
la = Laplace(model, 'regression', subset_of_weights='all', hessian_structure='full')
la.fit(train_loader)

# Otimização da verossimilhança marginal sobre os hiperparâmetros
log_prior = torch.ones(1, requires_grad=True)
log_sigma = torch.ones(1, requires_grad=True)
hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)

for epoch in range(n_epochs):
    hyper_optimizer.zero_grad()
    neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()

# Fazer previsões
x = X_test.flatten().cpu().numpy()

# Obter predições do modelo de Laplace
f_mu, f_var = la(X_test)

# Converter tensores para arrays NumPy
f_mu = f_mu.squeeze().detach().cpu().numpy()
f_var = f_var.squeeze().detach().cpu().numpy()
sigma_noise = la.sigma_noise.item()

# Calcular a variância total predita
pred_var = f_var + sigma_noise**2
pred_std = np.sqrt(pred_var)

# Converter dados de treinamento para arrays NumPy
X_train_np = X_train.cpu().numpy()
y_train_np = y_train.cpu().numpy()

# Plotar os resultados e salvar com o nome "cubic_example_plot_random_spacing"
plot_regression(
    X_train_np,
    y_train_np,
    x,
    f_mu,
    pred_std,
    file_name="cubic_example_plot_random_spacing",
    plot=False
)

# Exibir valores finais de prior e sigma
print("Valor final do prior:", log_prior.exp().item())
print("Valor final do sigma:", log_sigma.exp().item())
