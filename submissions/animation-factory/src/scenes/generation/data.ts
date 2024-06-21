import npyjs from 'npyjs';

const npyjsLoader = new npyjs();

const loadData = async (path: string) => {
  const data = await npyjsLoader.load(path);
  return data;
}