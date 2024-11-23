from laplace import Laplace
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(711)
# Modelo simples
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(1, 50)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(50, 1)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        return self.fc2(x)

# Dados de treinamento com buracos no meio (sem amostra no ponto 0)
def get_cubic_example_with_gaps_no_center(n_low, n_high, sigma_noise):
    """
    Gera dados de treinamento com buracos no meio, sem amostras no centro.
    `n_low` são os pontos concentrados nas extremidades.
    `n_high` define a densidade de pontos próximos aos máximos/mínimos.
    """
    # Amostras próximas aos extremos (-3, -2.5) e (2.5, 3)
    X_low = -3 + 0.5 * torch.rand(n_low).reshape(-1, 1)  # Região [-3, -2.5]
    X_high = 2.5 + 0.5 * torch.rand(n_high).reshape(-1, 1)  # Região [2.5, 3]

    # Combinar as regiões (sem adicionar ponto no meio)
    X_train = torch.cat([X_low, X_high], dim=0)

    # Gerar valores de saída (ruído adicionado)
    y_train = X_train**3 + sigma_noise * torch.randn(X_train.shape)

    # Conjunto de teste para todo o intervalo [-4, 4]
    X_test = torch.linspace(-4, 4, 100).reshape(-1, 1)

    # Criar DataLoader
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=len(X_train), shuffle=True)

    return X_train, y_train, train_loader, X_test

# Configurar os dados com buracos sem o ponto central
X_train, y_train, train_loader, X_test = get_cubic_example_with_gaps_no_center(
    n_low=10,  # 10 pontos próximos ao mínimo
    n_high=10,  # 10 pontos próximos ao máximo
    sigma_noise=0.3  # Ruído
)


# Inicializar modelo
model = SimpleNN()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Treinar modelo
n_epochs = 1000
for epoch in range(n_epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Laplace Approximation com otimização de prior_precision usando marglik
la = Laplace(model, 'regression', subset_of_weights='last_layer', hessian_structure='kron')
la.fit(train_loader)

# Otimizar prior_precision usando Marglik
print("\n--- Otimização da Prior Precision com Marglik ---")
la.optimize_prior_precision(method='marglik', pred_type='glm')

# Gerar dados de teste
X_test = torch.linspace(-5, 5, 100).reshape(-1, 1)

# Fazer previsões com incerteza
f_mu, f_var = la(X_test)
f_mu = f_mu.detach().numpy()
f_sigma = f_var.sqrt().detach().numpy()

# Certifique-se de que os valores sejam unidimensionais
f_mu = f_mu.squeeze()  # Remover dimensões extras
f_sigma = f_sigma.squeeze()  # Remover dimensões extras

# Plotar os resultados
plt.figure(figsize=(10, 6))
plt.scatter(X_train.numpy(), y_train.numpy(), color='blue', label='Dados de Treinamento')
plt.plot(X_test.numpy(), f_mu, color='red', label='Predição')
plt.fill_between(
    X_test.numpy().flatten(), 
    (f_mu - 2 * f_sigma), 
    (f_mu + 2 * f_sigma),
    color='orange', alpha=0.3, label='Incerteza (2σ)'
)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title("Laplace Approximation com Marglik 0 off")
plt.show()

# Exibir os valores da precisão a priori e sigma_noise após otimização
print("Valor otimizado da Prior Precision:", la.prior_precision_diag.cpu().detach().numpy())
print("Valor otimizado do Sigma Noise:", la.sigma_noise.item())

# Salvar o gráfico
file_name = "grafico_marglik0.png"
plt.savefig(file_name)
print(f"Gráfico salvo como {file_name}")
