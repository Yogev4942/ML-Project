from ucimlrepo import fetch_ucirepo
from model import model

def main():
    automobile = fetch_ucirepo(id=10)

    # Keep each engine-size example paired with its price; remove rows missing either value.
    data = automobile.data.features[["engine-size"]].join(
        automobile.data.targets[["price"]]
    ).dropna()

    # Scale values so gradient descent can learn with the model's current learning rate.
    engine_size = data["engine-size"].to_numpy(dtype=float) / 100
    price = data["price"].to_numpy(dtype=float) / 50000

    trainer = model()
    trainer.fit(engine_size, price, epochs=1000)

    predictions = trainer.predict(engine_size)
    print(f"Training examples: {len(engine_size)}")
    print(f"Training MSE (scaled): {trainer.MSE(price, predictions):.6f}")
    print(f"First predicted price: ${predictions[0] * 50000:.2f}")


if __name__ == "__main__":
    main()